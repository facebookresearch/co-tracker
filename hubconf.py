# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

_COTRACKER_URL = "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth"


def _make_cotracker_predictor(*, pretrained: bool = True, online=False, **kwargs):
    if online:
        from cotracker.predictor import CoTrackerOnlinePredictor

        predictor = CoTrackerOnlinePredictor(checkpoint=None)
    else:
        from cotracker.predictor import CoTrackerPredictor

        predictor = CoTrackerPredictor(checkpoint=None)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_COTRACKER_URL, map_location="cpu")
        predictor.model.load_state_dict(state_dict)
    return predictor


def cotracker2(*, pretrained: bool = True, **kwargs):
    """
    CoTracker2 with stride 4 and window length 8. Can track up to 265*265 points jointly.
    """
    return _make_cotracker_predictor(pretrained=pretrained, online=False, **kwargs)


def cotracker2_online(*, pretrained: bool = True, **kwargs):
    """
    Online CoTracker2 with stride 4 and window length 8. Can track up to 265*265 points jointly.
    """
    return _make_cotracker_predictor(pretrained=pretrained, online=True, **kwargs)
