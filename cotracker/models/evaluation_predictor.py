# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from typing import Tuple

from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.model_utils import (
    get_points_on_a_grid,
    get_uniformly_sampled_pts,
    get_sift_sampled_pts,
)
import numpy as np
import sys

from torchvision.transforms import Compose
from tqdm import tqdm
from cotracker.models.core.model_utils import bilinear_sampler


class EvaluationPredictor(torch.nn.Module):
    def __init__(
        self,
        cotracker_model: CoTrackerThreeOffline,
        interp_shape: Tuple[int, int] = (384, 512),
        grid_size: int = 5,
        local_grid_size: int = 8,
        single_point: bool = True,
        sift_size: int = 0,
        num_uniformly_sampled_pts: int = 0,
        n_iters: int = 6,
        local_extent: int = 50,
    ) -> None:
        super(EvaluationPredictor, self).__init__()
        self.grid_size = grid_size
        self.local_grid_size = local_grid_size
        self.sift_size = sift_size
        self.single_point = single_point
        self.interp_shape = interp_shape
        self.n_iters = n_iters
        self.num_uniformly_sampled_pts = num_uniformly_sampled_pts
        self.model = cotracker_model
        self.local_extent = local_extent
        self.model.eval()

    def forward(self, video, queries):
        queries = queries.clone()
        B, T, C, H, W = video.shape
        B, N, D = queries.shape

        assert D == 3
        assert B == 1
        interp_shape = self.interp_shape

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(
            video, tuple(interp_shape), mode="bilinear", align_corners=True
        )
        video = video.reshape(B, T, 3, interp_shape[0], interp_shape[1])

        device = video.device

        queries[:, :, 1] *= (interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (interp_shape[0] - 1) / (H - 1)

        if self.single_point:
            traj_e = torch.zeros((B, T, N, 2), device=device)
            vis_e = torch.zeros((B, T, N), device=device)
            conf_e = torch.zeros((B, T, N), device=device)

            for pind in range((N)):
                query = queries[:, pind : pind + 1]
                t = query[0, 0, 0].long()
                start_ind = 0
                traj_e_pind, vis_e_pind, conf_e_pind = self._process_one_point(
                    video[:,start_ind:], query
                )
                traj_e[:, start_ind:, pind : pind + 1] = traj_e_pind[:, :, :1]
                vis_e[:, start_ind:, pind : pind + 1] = vis_e_pind[:, :, :1]
                conf_e[:, start_ind:, pind : pind + 1] = conf_e_pind[:, :, :1]
        else:
            if self.grid_size > 0:
                xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
                xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(
                    device
                )  #
                queries = torch.cat([queries, xy], dim=1)  #

            if self.num_uniformly_sampled_pts > 0:
                xy = get_uniformly_sampled_pts(
                    self.num_uniformly_sampled_pts,
                    video.shape[1],
                    video.shape[3:],
                    device=device,
                )
                queries = torch.cat([queries, xy], dim=1)  #

            sift_size = self.sift_size
            if sift_size > 0:
                xy = get_sift_sampled_pts(video, sift_size, T, [H, W], device=device)
                if xy.shape[1] == sift_size:
                    queries = torch.cat([queries, xy], dim=1)  #
                else:
                    sift_size = 0

            preds = self.model(video=video, queries=queries, iters=self.n_iters)
            traj_e, vis_e = preds[0], preds[1]
            conf_e = None
            if len(preds) > 3:
                conf_e = preds[2]
            if (
                sift_size > 0
                or self.grid_size > 0
                or self.num_uniformly_sampled_pts > 0
            ):
                traj_e = traj_e[
                    :,
                    :,
                    : -self.grid_size**2 - sift_size - self.num_uniformly_sampled_pts,
                ]
                vis_e = vis_e[
                    :,
                    :,
                    : -self.grid_size**2 - sift_size - self.num_uniformly_sampled_pts,
                ]
                if conf_e is not None:
                    conf_e = conf_e[
                        :,
                        :,
                        : -self.grid_size**2
                        - sift_size
                        - self.num_uniformly_sampled_pts,
                    ]

        traj_e[:, :, :, 0] *= (W - 1) / float(interp_shape[1] - 1)
        traj_e[:, :, :, 1] *= (H - 1) / float(interp_shape[0] - 1)
        if conf_e is not None:
            vis_e = vis_e * conf_e

        return traj_e, vis_e

    def _process_one_point(self, video, query):
        t = query[0, 0, 0].long()
        B, T, C, H, W = video.shape
        device = query.device
        if self.local_grid_size > 0:
            xy_target = get_points_on_a_grid(
                self.local_grid_size,
                (self.local_extent, self.local_extent),
                [query[0, 0, 2].item(), query[0, 0, 1].item()],
            )

            xy_target = torch.cat(
                [torch.zeros_like(xy_target[:, :, :1]), xy_target], dim=2
            ).to(
                device
            )  #
            query = torch.cat([query, xy_target], dim=1)  #

        if self.grid_size > 0:
            xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
            xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #
            query = torch.cat([query, xy], dim=1)  #

        sift_size = self.sift_size
        if sift_size > 0:
            xy = get_sift_sampled_pts(video, sift_size, T, [H, W], device=device)
            sift_size = xy.shape[1]
            if sift_size > 0:
                query = torch.cat([query, xy], dim=1)  #

            num_uniformly_sampled_pts = self.sift_size - sift_size
            if num_uniformly_sampled_pts > 0:
                xy2 = get_uniformly_sampled_pts(
                    num_uniformly_sampled_pts,
                    video.shape[1],
                    video.shape[3:],
                    device=device,
                )
                query = torch.cat([query, xy2], dim=1)  #

        if self.num_uniformly_sampled_pts > 0:
            xy = get_uniformly_sampled_pts(
                self.num_uniformly_sampled_pts,
                video.shape[1],
                video.shape[3:],
                device=device,
            )
            query = torch.cat([query, xy], dim=1)  #

        traj_e_pind, vis_e_pind, conf_e_pind, __ = self.model(
            video=video, queries=query, iters=self.n_iters
        )

        return traj_e_pind[..., :2], vis_e_pind, conf_e_pind
