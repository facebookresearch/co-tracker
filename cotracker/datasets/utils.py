# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(eq=False)
class CoTrackerData:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    segmentation: torch.Tensor  # B, S, 1, H, W
    trajectory: torch.Tensor  # B, S, N, 2
    visibility: torch.Tensor  # B, S, N
    # optional data
    valid: Optional[torch.Tensor] = None  # B, S, N
    seq_name: Optional[str] = None
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format


def collate_fn(batch):
    """
    Collate function for video tracks data.
    """
    video = torch.stack([b.video for b in batch], dim=0)
    segmentation = torch.stack([b.segmentation for b in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b in batch], dim=0)
    visibility = torch.stack([b.visibility for b in batch], dim=0)
    query_points = None
    if batch[0].query_points is not None:
        query_points = torch.stack([b.query_points for b in batch], dim=0)
    seq_name = [b.seq_name for b in batch]

    return CoTrackerData(
        video,
        segmentation,
        trajectory,
        visibility,
        seq_name=seq_name,
        query_points=query_points,
    )


def collate_fn_train(batch):
    """
    Collate function for video tracks data during training.
    """
    gotit = [gotit for _, gotit in batch]
    video = torch.stack([b.video for b, _ in batch], dim=0)
    segmentation = torch.stack([b.segmentation for b, _ in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b, _ in batch], dim=0)
    visibility = torch.stack([b.visibility for b, _ in batch], dim=0)
    valid = torch.stack([b.valid for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    return (
        CoTrackerData(video, segmentation, trajectory, visibility, valid, seq_name),
        gotit,
    )


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj


def resize_sample(rgbs, trajs_g, segs, interp_shape):
    S, C, H, W = rgbs.shape
    S, N, D = trajs_g.shape

    assert D == 2

    rgbs = F.interpolate(rgbs, interp_shape, mode="bilinear")
    segs = F.interpolate(segs, interp_shape, mode="nearest")

    trajs_g[:, :, 0] *= interp_shape[1] / W
    trajs_g[:, :, 1] *= interp_shape[0] / H
    return rgbs, trajs_g, segs
