# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from cotracker.models.core.cotracker.cotracker import CoTracker2


def build_cotracker(
    checkpoint: str,
):
    if checkpoint is None:
        return build_cotracker()
    model_name = checkpoint.split("/")[-1].split(".")[0]
    if model_name == "cotracker":
        return build_cotracker(checkpoint=checkpoint)
    else:
        raise ValueError(f"Unknown model name {model_name}")


def build_cotracker(checkpoint=None):
    cotracker = CoTracker2(stride=4, window_len=8, add_space_attn=True)

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        cotracker.load_state_dict(state_dict)
    return cotracker
