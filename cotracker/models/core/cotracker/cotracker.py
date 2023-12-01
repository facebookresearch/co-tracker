# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from einops import rearrange

from cotracker.models.core.cotracker.blocks import (
    BasicEncoder,
    CorrBlock,
    UpdateFormer,
)

from cotracker.models.core.model_utils import meshgrid2d, bilinear_sample2d, smart_cat
from cotracker.models.core.embeddings import (
    get_2d_embedding,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)


torch.manual_seed(0)


def get_points_on_a_grid(grid_size, interp_shape, grid_center=(0, 0), device="cpu"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy


def sample_pos_embed(grid_size, embed_dim, coords):
    pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size)
    pos_embed = (
        torch.from_numpy(pos_embed)
        .reshape(grid_size[0], grid_size[1], embed_dim)
        .float()
        .unsqueeze(0)
        .to(coords.device)
    )
    sampled_pos_embed = bilinear_sample2d(
        pos_embed.permute(0, 3, 1, 2), coords[:, 0, :, 0], coords[:, 0, :, 1]
    )
    return sampled_pos_embed


class CoTracker(nn.Module):
    def __init__(
        self,
        S=8,
        stride=8,
        add_space_attn=True,
        num_heads=8,
        hidden_size=384,
        space_depth=12,
        time_depth=12,
    ):
        super(CoTracker, self).__init__()
        self.S = S
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = latent_dim = 128
        self.corr_levels = 4
        self.corr_radius = 3
        self.add_space_attn = add_space_attn
        self.fnet = BasicEncoder(
            output_dim=self.latent_dim, norm_fn="instance", dropout=0, stride=stride
        )

        self.updateformer = UpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=456,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
        )

        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

    def forward_iteration(
        self,
        fmaps,
        coords_init,
        feat_init=None,
        vis_init=None,
        track_mask=None,
        iters=4,
    ):
        # Get the shape of the points coordinates tensor
        # - TODO: why enforce D to 2? what is D?
        # - enforce batch size of 1 only
        B, S_init, N, D = coords_init.shape
        assert D == 2
        assert B == 1

        # Get the shape of the feature maps teensor
        B, S, __, H8, W8 = fmaps.shape

        device = fmaps.device

        # If we are at the beginning of the video (say, we have window length of 8, and we are starting at frame 4)...
        # then #TODO understand wtf this is
        if S_init < S:
            coords = torch.cat(
                [coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            vis_init = torch.cat(
                [vis_init, vis_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
        else:
            coords = coords_init.clone()

        # 
        fcorr_fn = CorrBlock(
            fmaps, num_levels=self.corr_levels, radius=self.corr_radius
        )

        
        ffeats = feat_init.clone()

        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1)

        pos_embed = sample_pos_embed(
            grid_size=(H8, W8),
            embed_dim=456,
            coords=coords,
        )
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)
        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(456, times_[0]))[None]
            .repeat(B, 1, 1)
            .float()
            .to(device)
        )
        coord_predictions = []

        for __ in range(iters):
            coords = coords.detach()
            fcorr_fn.corr(ffeats)

            fcorrs = fcorr_fn.sample(coords)  # B, S, N, LRR
            LRR = fcorrs.shape[3]

            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)

            flows_cat = get_2d_embedding(flows_, 64, cat_coords=True)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0]).repeat(
                            1, vis_init.shape[1] - track_mask.shape[1], 1, 1
                        ),
                    ],
                    dim=1,
                )
            concat = (
                torch.cat([track_mask, vis_init], dim=2)
                .permute(0, 2, 1, 3)
                .reshape(B * N, S, 2)
            )

            transformer_input = torch.cat([flows_cat, fcorrs_, ffeats_, concat], dim=2)
            x = transformer_input + pos_embed + times_embed

            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            delta = self.updateformer(x)

            delta = rearrange(delta, " b n t d -> (b n) t d")

            delta_coords_ = delta[:, :, :2]
            delta_feats_ = delta[:, :, 2:]

            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)

            ffeats_ = self.ffeat_updater(self.norm(delta_feats_)) + ffeats_

            ffeats = ffeats_.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C

            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)
            coord_predictions.append(coords * self.stride)

        vis_e = self.vis_predictor(ffeats.reshape(B * S * N, self.latent_dim)).reshape(
            B, S, N
        )
        return coord_predictions, vis_e, feat_init


    def forward(self, 
                rgbs,               # this is the input sequence of frames
                queries,            # this is the points we want to track
                iters=4,            # number of transformer iterations
                feat_init=None,     
                is_train=False      # whether train or eval
            ):
        B, T, C, H, W = rgbs.shape  # Batch, Time, Channels, Height, Width
        B, N, __ = queries.shape    # Batch, Number of points, 3 (x, y, visibility)

        device = rgbs.device        # device (cpu or gpu)
        assert B == 1               # enforcing a batch size of one
        
        # INIT for the first sequence
        # We want to sort points by the first frame they are visible to add them to the tensor of tracked points consecutively
        first_positive_inds = queries[:, :, 0].long()

        __, sort_inds = torch.sort(first_positive_inds[0], dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[0][sort_inds]

        assert torch.allclose(
            first_positive_inds[0], first_positive_inds[0][sort_inds][inv_sort_inds]
        )

        coords_init = queries[:, :, 1:].reshape(B, 1, N, 2).repeat(
            1, self.S, 1, 1
        ) / float(self.stride)

        rgbs = 2 * (rgbs / 255.0) - 1.0

        traj_e = torch.zeros((B, T, N, 2), device=device)
        vis_e = torch.zeros((B, T, N), device=device)

        ind_array = torch.arange(T, device=device)
        ind_array = ind_array[None, :, None].repeat(B, 1, N)

        track_mask = (ind_array >= first_positive_inds[:, None, :]).unsqueeze(-1)
        # these are logits, so we initialize visibility with something that would give a value close to 1 after softmax
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10

        ind = 0

        track_mask_ = track_mask[:, :, sort_inds].clone()
        coords_init_ = coords_init[:, :, sort_inds].clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()

        prev_wind_idx = 0
        fmaps_ = None
        vis_predictions = []
        coord_predictions = []
        wind_inds = []

        # END OF SETUP
        # START OF INFERENCE

        # We iterate over the sequence of frames until the start index of the window is greater than the last frame - (window size / 2)
        while ind < T - self.S // 2:

            # extract the sequence of frames based on the current index
            rgbs_seq = rgbs[:, ind : ind + self.S]

            # Window Size Adjustment: This block deals with the window size (self.S) of the video frames being processed. 
            # If the current sequence (rgbs_seq) has fewer frames than the predefined window size, it is padded to match this size.
            # Padding Mechanism: Padding is done by repeating the last frame of the sequence (rgbs_seq[:, -1, None]) until the
            # sequence length equals the window size. This repetition is necessary to maintain a consistent input size for further processing.
            S = S_local = rgbs_seq.shape[1]
            if S < self.S:
                rgbs_seq = torch.cat(
                    [rgbs_seq, rgbs_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                    dim=1,
                )
                S = rgbs_seq.shape[1]
            
            # Reshaping for CNN Processing: The reshaped rgbs_ tensor prepares the input for the convolutional neural network (CNN). 
            # By reshaping the sequence into (B * S, C, H, W), each frame in the sequence is treated as a separate input for feature extraction,
            # where B is the batch size, S is the sequence length (window size), C is the number of channels, H is the height, and W is the width of the frame.
            rgbs_ = rgbs_seq.reshape(B * S, C, H, W)

            # Initial Feature Map Extraction: 
            # When processing the first window (fmaps_ is None), 
            # the feature maps are extracted for the entire window using the feature extraction network (self.fnet
            if fmaps_ is None:
                fmaps_ = self.fnet(rgbs_)

            # For subsequent windows, feature maps from the new half of the window (rgbs_[self.S // 2 :])
            # are extracted and concatenated with the latter half of the previously computed feature maps (fmaps_[self.S // 2 :]). 
            # This ensures that there's a continuity in the feature representation as the window slides over the video.
            else:
                fmaps_ = torch.cat(
                    [fmaps_[self.S // 2 :], self.fnet(rgbs_[self.S // 2 :])], dim=0
                )

            # Reshaping for Transformer Processing: The reshaped fmaps tensor arranges the feature maps to be compatible 
            # with the transformer network. It groups the feature maps by batch, sequence, and feature dimensions
            fmaps = fmaps_.reshape(
                B, S, self.latent_dim, H // self.stride, W // self.stride
            )

            # Initialize a placeholder for current window points.
            # we are gettting the indices of points where the point is first visible at or before the end of the current window
            curr_wind_points = torch.nonzero(first_positive_sorted_inds < ind + self.S)

            # if none of the points are yet visible, skip rest of floop
            if curr_wind_points.shape[0] == 0:
                ind = ind + self.S // 2
                continue

            # identify next visibile points
            wind_idx = curr_wind_points[-1] + 1

            # Check if any points have become visible in this window compared to last window
            # If new points have appeared, their features need to be initialized.
            if wind_idx - prev_wind_idx > 0:
                # extracts the relevant feature maps for the newly visible points. This step is crucial as it aligns the feature maps with the spatial locations of these points.
                fmaps_sample = fmaps[
                    :, first_positive_sorted_inds[prev_wind_idx:wind_idx] - ind
                ]

                # Initializing Features Using Bilinear Sampling:
                feat_init_ = bilinear_sample2d(
                    fmaps_sample,
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 0],
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 1],
                ).permute(0, 2, 1)

                # Preparing Features for Transformer Input
                # essentially replicates the initial feature vector across all frames in the window, as the transformer requires consistent input dimensions.
                feat_init_ = feat_init_.unsqueeze(1).repeat(1, self.S, 1, 1)
                feat_init = smart_cat(feat_init, feat_init_, dim=2)

            # TODO: undnerstand this chunk
            if prev_wind_idx > 0:
                new_coords = coords[-1][:, self.S // 2 :] / float(self.stride)

                coords_init_[:, : self.S // 2, :prev_wind_idx] = new_coords
                coords_init_[:, self.S // 2 :, :prev_wind_idx] = new_coords[
                    :, -1
                ].repeat(1, self.S // 2, 1, 1)

                new_vis = vis[:, self.S // 2 :].unsqueeze(-1)
                vis_init_[:, : self.S // 2, :prev_wind_idx] = new_vis
                vis_init_[:, self.S // 2 :, :prev_wind_idx] = new_vis[:, -1].repeat(
                    1, self.S // 2, 1, 1
                )

            # Transformer iterations -- the "main inference" is here
            coords, vis, __ = self.forward_iteration(
                fmaps=fmaps,
                coords_init=coords_init_[:, :, :wind_idx],
                feat_init=feat_init[:, :, :wind_idx],
                vis_init=vis_init_[:, :, :wind_idx],
                track_mask=track_mask_[:, ind : ind + self.S, :wind_idx],
                iters=iters,
            )
            if is_train:
                vis_predictions.append(torch.sigmoid(vis[:, :S_local]))
                coord_predictions.append([coord[:, :S_local] for coord in coords])
                wind_inds.append(wind_idx)


            traj_e[:, ind : ind + self.S, :wind_idx] = coords[-1][:, :S_local]
            vis_e[:, ind : ind + self.S, :wind_idx] = vis[:, :S_local]

            track_mask_[:, : ind + self.S, :wind_idx] = 0.0
            ind = ind + self.S // 2

            prev_wind_idx = wind_idx

        traj_e = traj_e[:, :, inv_sort_inds]
        vis_e = vis_e[:, :, inv_sort_inds]

        vis_e = torch.sigmoid(vis_e)

        train_data = (
            (vis_predictions, coord_predictions, wind_inds, sort_inds)
            if is_train
            else None
        )
        return traj_e, feat_init, vis_e, train_data
