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


def get_points_on_a_grid(grid_size, interp_shape, grid_center=(0, 0), device="cuda"):
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
        B, S_init, N, D = coords_init.shape
        assert D == 2
        assert B == 1

        B, S, __, H8, W8 = fmaps.shape

        device = fmaps.device

        if S_init < S:
            coords = torch.cat(
                [coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            vis_init = torch.cat(
                [vis_init, vis_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
        else:
            coords = coords_init.clone()

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

    def forward(self, rgbs, queries, iters=4, feat_init=None, is_train=False):
        B, T, C, H, W = rgbs.shape
        B, N, __ = queries.shape

        device = rgbs.device
        assert B == 1
        # INIT for the first sequence
        # We want to sort points by the first frame they are visible to add them to the tensor of tracked points consequtively
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
        while ind < T - self.S // 2:
            rgbs_seq = rgbs[:, ind : ind + self.S]

            S = S_local = rgbs_seq.shape[1]
            if S < self.S:
                rgbs_seq = torch.cat(
                    [rgbs_seq, rgbs_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                    dim=1,
                )
                S = rgbs_seq.shape[1]
            rgbs_ = rgbs_seq.reshape(B * S, C, H, W)

            if fmaps_ is None:
                fmaps_ = self.fnet(rgbs_)
            else:
                fmaps_ = torch.cat(
                    [fmaps_[self.S // 2 :], self.fnet(rgbs_[self.S // 2 :])], dim=0
                )
            fmaps = fmaps_.reshape(
                B, S, self.latent_dim, H // self.stride, W // self.stride
            )

            curr_wind_points = torch.nonzero(first_positive_sorted_inds < ind + self.S)
            if curr_wind_points.shape[0] == 0:
                ind = ind + self.S // 2
                continue
            wind_idx = curr_wind_points[-1] + 1

            if wind_idx - prev_wind_idx > 0:
                fmaps_sample = fmaps[
                    :, first_positive_sorted_inds[prev_wind_idx:wind_idx] - ind
                ]

                feat_init_ = bilinear_sample2d(
                    fmaps_sample,
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 0],
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 1],
                ).permute(0, 2, 1)

                feat_init_ = feat_init_.unsqueeze(1).repeat(1, self.S, 1, 1)
                feat_init = smart_cat(feat_init, feat_init_, dim=2)

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
