# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from typing import Tuple

from cotracker.models.core.cotracker.cotracker import CoTracker2
from cotracker.models.core.model_utils import get_points_on_a_grid


class EvaluationPredictor(torch.nn.Module):
    def __init__(
        self,
        cotracker_model: CoTracker2,
        interp_shape: Tuple[int, int] = (384, 512),
        grid_size: int = 5,
        local_grid_size: int = 8,
        single_point: bool = True,
        n_iters: int = 6,
    ) -> None:
        super(EvaluationPredictor, self).__init__()
        self.grid_size = grid_size
        self.local_grid_size = local_grid_size
        self.single_point = single_point
        self.interp_shape = interp_shape
        self.n_iters = n_iters

        self.model = cotracker_model
        self.model.eval()

    def forward(self, video, queries):
        queries = queries.clone()
        B, T, C, H, W = video.shape
        B, N, D = queries.shape

        assert D == 3

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        device = video.device

        queries[:, :, 1] *= (self.interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (self.interp_shape[0] - 1) / (H - 1)

        if self.single_point:
            traj_e = torch.zeros((B, T, N, 2), device=device)
            vis_e = torch.zeros((B, T, N), device=device)
            for pind in range((N)):
                query = queries[:, pind : pind + 1]

                t = query[0, 0, 0].long()

                traj_e_pind, vis_e_pind = self._process_one_point(video, query)
                traj_e[:, t:, pind : pind + 1] = traj_e_pind[:, :, :1]
                vis_e[:, t:, pind : pind + 1] = vis_e_pind[:, :, :1]
        else:
            if self.grid_size > 0:
                xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
                xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #
                queries = torch.cat([queries, xy], dim=1)  #

            traj_e, vis_e, __ = self.model(
                video=video,
                queries=queries,
                iters=self.n_iters,
            )

        traj_e[:, :, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        traj_e[:, :, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)
        return traj_e, vis_e

    def _process_one_point(self, video, query):
        t = query[0, 0, 0].long()

        device = query.device
        if self.local_grid_size > 0:
            xy_target = get_points_on_a_grid(
                self.local_grid_size,
                (50, 50),
                [query[0, 0, 2].item(), query[0, 0, 1].item()],
            )

            xy_target = torch.cat([torch.zeros_like(xy_target[:, :, :1]), xy_target], dim=2).to(
                device
            )  #
            query = torch.cat([query, xy_target], dim=1)  #

        if self.grid_size > 0:
            xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
            xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #
            query = torch.cat([query, xy], dim=1)  #
        # crop the video to start from the queried frame
        query[0, 0, 0] = 0
        traj_e_pind, vis_e_pind, __ = self.model(
            video=video[:, t:], queries=query, iters=self.n_iters
        )

        return traj_e_pind, vis_e_pind
