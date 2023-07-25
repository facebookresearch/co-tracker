# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from cotracker.models.core.cotracker.cotracker import CoTracker


def build_cotracker(
    checkpoint: str,
):
    if checkpoint is None:
        return build_cotracker_stride_4_wind_8()
    model_name = checkpoint.split("/")[-1].split(".")[0]
    if model_name == "cotracker_stride_4_wind_8":
        return build_cotracker_stride_4_wind_8(checkpoint=checkpoint)
    elif model_name == "cotracker_stride_4_wind_12":
        return build_cotracker_stride_4_wind_12(checkpoint=checkpoint)
    elif model_name == "cotracker_stride_8_wind_16":
        return build_cotracker_stride_8_wind_16(checkpoint=checkpoint)
    else:
        raise ValueError(f"Unknown model name {model_name}")


# model used to produce the results in the paper
def build_cotracker_stride_4_wind_8(checkpoint=None):
    return _build_cotracker(
        stride=4,
        sequence_len=8,
        checkpoint=checkpoint,
    )


def build_cotracker_stride_4_wind_12(checkpoint=None):
    return _build_cotracker(
        stride=4,
        sequence_len=12,
        checkpoint=checkpoint,
    )


# the fastest model
def build_cotracker_stride_8_wind_16(checkpoint=None):
    return _build_cotracker(
        stride=8,
        sequence_len=16,
        checkpoint=checkpoint,
    )


def _build_cotracker(
    stride,
    sequence_len,
    checkpoint=None,
):
    cotracker = CoTracker(
        stride=stride,
        S=sequence_len,
        add_space_attn=True,
        space_depth=6,
        time_depth=6,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        cotracker.load_state_dict(state_dict)
    return cotracker
