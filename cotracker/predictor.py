# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid
from cotracker.models.build_cotracker import build_cotracker


class CoTrackerPredictor(torch.nn.Module):
    def __init__(
        self,
        checkpoint="./checkpoints/scaled_offline.pth",
        offline=True,
        v2=False,
        window_len=60,
    ):
        super().__init__()
        self.v2 = v2
        self.support_grid_size = 6
        model = build_cotracker(
            checkpoint,
            v2=v2,
            offline=offline,
            window_len=window_len,
        )
        self.interp_shape = model.model_resolution
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video,  # (B, T, 3, H, W)
        # input prompt types:
        # - None. Dense tracks are computed in this case. You can adjust *query_frame* to compute tracks starting from a specific frame.
        # *backward_tracking=True* will compute tracks in both directions.
        # - queries. Queried points of shape (B, N, 3) in format (t, x, y) for frame index and pixel coordinates.
        # - grid_size. Grid of N*N points from the first frame. if segm_mask is provided, then computed only for the mask.
        # You can adjust *query_frame* and *backward_tracking* for the regular grid in the same way as for dense tracks.
        queries: torch.Tensor = None,
        segm_mask: torch.Tensor = None,  # Segmentation mask of shape (B, 1, H, W)
        grid_size: int = 0,
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        backward_tracking: bool = False,
    ):
        if queries is None and grid_size == 0:
            tracks, visibilities = self._compute_dense_tracks(
                video,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
            )
        else:
            tracks, visibilities = self._compute_sparse_tracks(
                video,
                queries,
                segm_mask,
                grid_size,
                add_support_grid=(grid_size == 0 or segm_mask is not None),
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
            )

        return tracks, visibilities

    def _compute_dense_tracks(
        self, video, grid_query_frame, grid_size=80, backward_tracking=False
    ):
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = None
        grid_pts = torch.zeros((video.shape[0], grid_width * grid_height, 3)).to(video.device)
        grid_pts[:, :, 0] = grid_query_frame
        for offset in range(grid_step * grid_step):
            print(f"step {offset} / {grid_step * grid_step}")
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[:, :, 1] = (
                torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            )
            grid_pts[:, :, 2] = (
                torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy
            )
            tracks_step, visibilities_step = self._compute_sparse_tracks(
                video=video,
                queries=grid_pts,
                backward_tracking=backward_tracking,
            )
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)

        return tracks, visibilities

    def _compute_sparse_tracks(
        self,
        video,
        queries,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
    ):
        B, T, C, H, W = video.shape

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(
            video, tuple(self.interp_shape), mode="bilinear", align_corners=True
        )
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        if queries is not None:
            B, N, D = queries.shape
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (self.interp_shape[1] - 1) / (W - 1),
                    (self.interp_shape[0] - 1) / (H - 1),
                ]
            )
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(
                grid_size, self.interp_shape, device=video.device
            )
            if segm_mask is not None:
                segm_mask = F.interpolate(
                    segm_mask, tuple(self.interp_shape), mode="nearest"
                )
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            ).repeat(B, 1, 1)

        if add_support_grid:
            grid_pts = get_points_on_a_grid(
                self.support_grid_size, self.interp_shape, device=video.device
            )
            grid_pts = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
            )
            grid_pts = grid_pts.repeat(B, 1, 1)
            queries = torch.cat([queries, grid_pts], dim=1)

        tracks, visibilities, *_ = self.model.forward(
            video=video, queries=queries, iters=6
        )

        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(
                video, queries, tracks, visibilities
            )
            if add_support_grid:
                queries[:, -self.support_grid_size**2 :, 0] = T - 1
        if add_support_grid:
            tracks = tracks[:, :, : -self.support_grid_size**2]
            visibilities = visibilities[:, :, : -self.support_grid_size**2]
        thr = 0.9
        visibilities = visibilities > thr

        # correct query-point predictions
        # see https://github.com/facebookresearch/co-tracker/issues/28

        # TODO: batchify
        for i in range(len(queries)):
            queries_t = queries[i, : tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, : tracks.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibilities[i, queries_t, arange] = True

        tracks *= tracks.new_tensor(
            [(W - 1) / (self.interp_shape[1] - 1), (H - 1) / (self.interp_shape[0] - 1)]
        )
        return tracks, visibilities

    def _compute_backward_tracks(self, video, queries, tracks, visibilities):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

        inv_tracks, inv_visibilities, *_ = self.model(
            video=inv_video, queries=inv_queries, iters=6
        )

        inv_tracks = inv_tracks.flip(1)
        inv_visibilities = inv_visibilities.flip(1)
        arange = torch.arange(video.shape[1], device=queries.device)[None, :, None]

        mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, 2)

        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        return tracks, visibilities


class CoTrackerOnlinePredictor(torch.nn.Module):
    def __init__(
        self,
        checkpoint="./checkpoints/scaled_online.pth",
        offline=False,
        v2=False,
        window_len=16,
    ):
        super().__init__()
        self.v2 = v2
        self.support_grid_size = 6
        model = build_cotracker(checkpoint, v2=v2, offline=False, window_len=window_len)
        self.interp_shape = model.model_resolution
        self.step = model.window_len // 2
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video_chunk,
        is_first_step: bool = False,
        queries: torch.Tensor = None,
        grid_size: int = 5,
        grid_query_frame: int = 0,
        add_support_grid=False,
    ):
        B, T, C, H, W = video_chunk.shape
        # Initialize online video processing and save queried points
        # This needs to be done before processing *each new video*
        if is_first_step:
            self.model.init_video_online_processing()
            if queries is not None:
                B, N, D = queries.shape
                self.N = N
                assert D == 3
                queries = queries.clone()
                queries[:, :, 1:] *= queries.new_tensor(
                    [
                        (self.interp_shape[1] - 1) / (W - 1),
                        (self.interp_shape[0] - 1) / (H - 1),
                    ]
                )
                if add_support_grid:
                    grid_pts = get_points_on_a_grid(
                        self.support_grid_size, self.interp_shape, device=video_chunk.device
                    )
                    grid_pts = torch.cat(
                        [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
                    )
                    queries = torch.cat([queries, grid_pts], dim=1)
            elif grid_size > 0:
                grid_pts = get_points_on_a_grid(
                    grid_size, self.interp_shape, device=video_chunk.device
                )
                self.N = grid_size**2
                queries = torch.cat(
                    [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                    dim=2,
                )
            
            self.queries = queries
            return (None, None)

        video_chunk = video_chunk.reshape(B * T, C, H, W)
        video_chunk = F.interpolate(
            video_chunk, tuple(self.interp_shape), mode="bilinear", align_corners=True
        )
        video_chunk = video_chunk.reshape(
            B, T, 3, self.interp_shape[0], self.interp_shape[1]
        )
        if self.v2:
            tracks, visibilities, __ = self.model(
                video=video_chunk, queries=self.queries, iters=6, is_online=True
            )
        else:
            tracks, visibilities, confidence, __ = self.model(
                video=video_chunk, queries=self.queries, iters=6, is_online=True
            )
        if add_support_grid:
            tracks = tracks[:,:,:self.N]
            visibilities = visibilities[:,:,:self.N]
            if not self.v2:
                confidence = confidence[:,:,:self.N]
            
        if not self.v2:
            visibilities = visibilities * confidence
        thr = 0.6
        return (
            tracks
            * tracks.new_tensor(
                [
                    (W - 1) / (self.interp_shape[1] - 1),
                    (H - 1) / (self.interp_shape[0] - 1),
                ]
            ),
            visibilities > thr,
        )
