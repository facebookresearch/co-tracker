# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from cotracker.models.core.model_utils import sample_features4d, sample_features5d
from cotracker.models.core.embeddings import (
    get_2d_embedding,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

from cotracker.models.core.cotracker.blocks import (
    Mlp,
    BasicEncoder,
    AttnBlock,
    CorrBlock,
    Attention,
)

torch.manual_seed(0)


class CoTracker2(nn.Module):
    def __init__(
        self,
        window_len=8,
        stride=4,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
    ):
        super(CoTracker2, self).__init__()
        self.window_len = window_len
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = 128
        self.add_space_attn = add_space_attn
        self.fnet = BasicEncoder(output_dim=self.latent_dim)
        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution
        self.input_dim = 456
        self.updateformer = EfficientUpdateFormer(
            space_depth=6,
            time_depth=6,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=self.latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            num_virtual_tracks=num_virtual_tracks,
        )

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(
            1, window_len, 1
        )

        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0])
        )

        self.register_buffer(
            "pos_emb",
            get_2d_sincos_pos_embed(
                embed_dim=self.input_dim,
                grid_size=(
                    model_resolution[0] // stride,
                    model_resolution[1] // stride,
                ),
            ),
        )
        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.track_feat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

    def forward_window(
        self,
        fmaps,
        coords,
        track_feat=None,
        vis=None,
        track_mask=None,
        attention_mask=None,
        iters=4,
    ):
        # B = batch size
        # S = number of frames in the window)
        # N = number of tracks
        # C = channels of a point feature vector
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # track_feat = B S N C
        # vis = B S N 1
        # track_mask = B S N 1
        # attention_mask = B S N

        B, S_init, N, __ = track_mask.shape
        B, S, *_ = fmaps.shape

        track_mask = F.pad(track_mask, (0, 0, 0, 0, 0, S - S_init), "constant")
        track_mask_vis = (
            torch.cat([track_mask, vis], dim=-1)
            .permute(0, 2, 1, 3)
            .reshape(B * N, S, 2)
        )

        corr_block = CorrBlock(
            fmaps,
            num_levels=4,
            radius=3,
            padding_mode="border",
        )

        sampled_pos_emb = (
            sample_features4d(self.pos_emb.repeat(B, 1, 1, 1), coords[:, 0])
            .reshape(B * N, self.input_dim)
            .unsqueeze(1)
        )  # B E N -> (B N) 1 E

        coord_preds = []
        for __ in range(iters):
            coords = coords.detach()  # B S N 2
            corr_block.corr(track_feat)

            # Sample correlation features around each point
            fcorrs = corr_block.sample(coords)  # (B N) S LRR

            # Get the flow embeddings
            flows = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)
            flow_emb = get_2d_embedding(flows, 64, cat_coords=True)  # N S E

            track_feat_ = track_feat.permute(0, 2, 1, 3).reshape(
                B * N, S, self.latent_dim
            )

            transformer_input = torch.cat(
                [flow_emb, fcorrs, track_feat_, track_mask_vis], dim=2
            )
            x = transformer_input + sampled_pos_emb + self.time_emb
            x = x.view(B, N, S, -1)  # (B N) S D -> B N S D

            delta = self.updateformer(
                x,
                attention_mask.reshape(B * S, N),  # B S N -> (B S) N
            )

            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            coords = coords + delta_coords
            coord_preds.append(coords * self.stride)

            delta_feats_ = delta[..., 2:].reshape(B * N * S, self.latent_dim)
            track_feat_ = track_feat.permute(0, 2, 1, 3).reshape(
                B * N * S, self.latent_dim
            )
            track_feat_ = self.track_feat_updater(self.norm(delta_feats_)) + track_feat_
            track_feat = track_feat_.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # (B N S) C -> B S N C

        vis_pred = self.vis_predictor(track_feat).reshape(B, S, N)
        return coord_preds, vis_pred

    def get_track_feat(self, fmaps, queried_frames, queried_coords):
        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat(
            [
                sample_frames,
                queried_coords[:, None],
            ],
            dim=-1,
        )
        sample_track_feats = sample_features5d(fmaps, sample_coords)
        return sample_track_feats

    def init_video_online_processing(self):
        self.online_ind = 0
        self.online_track_feat = None
        self.online_coords_predicted = None
        self.online_vis_predicted = None

    def forward(self, video, queries, iters=4, is_train=False, is_online=False):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
            is_online (bool, optional): enables online mode. Defaults to False. Before enabling, call model.init_video_online_processing().

        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """
        B, T, C, H, W = video.shape
        B, N, __ = queries.shape
        S = self.window_len
        device = queries.device

        # B = batch size
        # S = number of frames in the window of the padded video
        # S_trimmed = actual number of frames in the window
        # N = number of tracks
        # C = color channels (3 for RGB)
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # video = B T C H W
        # queries = B N 3
        # coords_init = B S N 2
        # vis_init = B S N 1

        assert S >= 2  # A tracker needs at least two frames to track something
        if is_online:
            assert T <= S, "Online mode: video chunk must be <= window size."
            assert (
                self.online_ind is not None
            ), "Call model.init_video_online_processing() first."
            assert not is_train, "Training not supported in online mode."
        step = S // 2  # How much the sliding window moves at every step
        video = 2 * (video / 255.0) - 1.0

        # The first channel is the frame number
        # The rest are the coordinates of points we want to track
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:]
        queried_coords = queried_coords / self.stride

        # We store our predictions here
        coords_predicted = torch.zeros((B, T, N, 2), device=device)
        vis_predicted = torch.zeros((B, T, N), device=device)
        if is_online:
            if self.online_coords_predicted is None:
                # Init online predictions with zeros
                self.online_coords_predicted = coords_predicted
                self.online_vis_predicted = vis_predicted
            else:
                # Pad online predictions with zeros for the current window
                pad = min(step, T - step)
                coords_predicted = F.pad(
                    self.online_coords_predicted, (0, 0, 0, 0, 0, pad), "constant"
                )
                vis_predicted = F.pad(
                    self.online_vis_predicted, (0, 0, 0, pad), "constant"
                )
        all_coords_predictions, all_vis_predictions = [], []

        # Pad the video so that an integer number of sliding windows fit into it
        # TODO: we may drop this requirement because the transformer should not care
        # TODO: pad the features instead of the video
        pad = (
            S - T if is_online else (S - T % S) % S
        )  # We don't want to pad if T % S == 0
        video = video.reshape(B, 1, T, C * H * W)
        video_pad = video[:, :, -1:].repeat(1, 1, pad, 1)
        video = torch.cat([video, video_pad], dim=2)

        # Compute convolutional features for the video or for the current chunk in case of online mode
        fmaps = self.fnet(video.reshape(-1, C, H, W)).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )

        # We compute track features
        track_feat = self.get_track_feat(
            fmaps,
            queried_frames - self.online_ind if is_online else queried_frames,
            queried_coords,
        ).repeat(1, S, 1, 1)
        if is_online:
            # We update track features for the current window
            sample_frames = queried_frames[:, None, :, None]  # B 1 N 1
            left = 0 if self.online_ind == 0 else self.online_ind + step
            right = self.online_ind + S
            sample_mask = (sample_frames >= left) & (sample_frames < right)
            if self.online_track_feat is None:
                self.online_track_feat = torch.zeros_like(track_feat, device=device)
            self.online_track_feat += track_feat * sample_mask
            track_feat = self.online_track_feat.clone()
        # We process ((num_windows - 1) * step + S) frames in total, so there are
        # (ceil((T - S) / step) + 1) windows
        num_windows = (T - S + step - 1) // step + 1
        # We process only the current video chunk in the online mode
        indices = [self.online_ind] if is_online else range(0, step * num_windows, step)

        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, S, N, 2).float()
        vis_init = torch.ones((B, S, N, 1), device=device).float() * 10
        for ind in indices:
            # We copy over coords and vis for tracks that are queried
            # by the end of the previous window, which is ind + overlap
            if ind > 0:
                overlap = S - step
                copy_over = (queried_frames < ind + overlap)[
                    :, None, :, None
                ]  # B 1 N 1
                coords_prev = torch.nn.functional.pad(
                    coords_predicted[:, ind : ind + overlap] / self.stride,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 2
                vis_prev = torch.nn.functional.pad(
                    vis_predicted[:, ind : ind + overlap, :, None].clone(),
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 1
                coords_init = torch.where(
                    copy_over.expand_as(coords_init), coords_prev, coords_init
                )
                vis_init = torch.where(
                    copy_over.expand_as(vis_init), vis_prev, vis_init
                )

            # The attention mask is 1 for the spatio-temporal points within
            # a track which is updated in the current window
            attention_mask = (
                (queried_frames < ind + S).reshape(B, 1, N).repeat(1, S, 1)
            )  # B S N

            # The track mask is 1 for the spatio-temporal points that actually
            # need updating: only after begin queried, and not if contained
            # in a previous window
            track_mask = (
                queried_frames[:, None, :, None]
                <= torch.arange(ind, ind + S, device=device)[None, :, None, None]
            ).contiguous()  # B S N 1

            if ind > 0:
                track_mask[:, :overlap, :, :] = False

            # Predict the coordinates and visibility for the current window
            coords, vis = self.forward_window(
                fmaps=fmaps if is_online else fmaps[:, ind : ind + S],
                coords=coords_init,
                track_feat=attention_mask.unsqueeze(-1) * track_feat,
                vis=vis_init,
                track_mask=track_mask,
                attention_mask=attention_mask,
                iters=iters,
            )

            S_trimmed = (
                T if is_online else min(T - ind, S)
            )  # accounts for last window duration
            coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed]
            vis_predicted[:, ind : ind + S] = vis[:, :S_trimmed]
            if is_train:
                all_coords_predictions.append(
                    [coord[:, :S_trimmed] for coord in coords]
                )
                all_vis_predictions.append(torch.sigmoid(vis[:, :S_trimmed]))

        if is_online:
            self.online_ind += step
            self.online_coords_predicted = coords_predicted
            self.online_vis_predicted = vis_predicted
        vis_predicted = torch.sigmoid(vis_predicted)

        if is_train:
            mask = (
                queried_frames[:, None]
                <= torch.arange(0, T, device=device)[None, :, None]
            )
            train_data = (all_coords_predictions, all_vis_predictions, mask)
        else:
            train_data = None

        return coords_predicted, vis_predicted, train_data


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        num_virtual_tracks=64,
        add_space_attn=True,
        linear_layer_for_vis_conf=False,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        if linear_layer_for_vis_conf:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)
        )
        self.add_space_attn = add_space_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                )
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        def _trunc_init(module):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None, add_space_attn=True):
        tokens = self.input_transform(input_tensor)

        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape
        j = 0
        layers = []
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if (
                add_space_attn
                and hasattr(self, "space_virtual_blocks")
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C

                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )

                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )  # (B T) N C -> B N T C
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]

        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow


class CrossAttnBlock(nn.Module):
    def __init__(
        self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, mask=None):
        attn_bias = None
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x
