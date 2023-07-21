# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

dependencies = ["torch", "einops", "timm", "tqdm"]

_COTRACKER_URL = (
    "https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth"
)


def _make_cotracker_predictor(*, pretrained: bool = True, **kwargs):
    from cotracker.predictor import CoTrackerPredictor

    predictor = CoTrackerPredictor(checkpoint=None)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            _COTRACKER_URL, map_location="cpu"
        )
        predictor.model.load_state_dict(state_dict)
    return predictor


def cotracker_w8(*, pretrained: bool = True, **kwargs):
    """
    CoTracker model with stride 4 and window length 8. (The main model from the paper)
    """
    return _make_cotracker_predictor(pretrained=pretrained, **kwargs)
