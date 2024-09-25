# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

_COTRACKER_URL = "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth"
_COTRACKER2v1_URL = "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2v1.pth"


def _make_cotracker_predictor(*, pretrained: bool = True, online=False, version="2", **kwargs):
    if online:
        from cotracker.predictor import CoTrackerOnlinePredictor

        if version == "2":
            predictor = CoTrackerOnlinePredictor(checkpoint=None)
        elif version == "2.1":
            predictor = CoTrackerOnlinePredictor(checkpoint=None, window_len=16)
    else:
        from cotracker.predictor import CoTrackerPredictor

        if version == "2":
            predictor = CoTrackerPredictor(checkpoint=None)
        elif version == "2.1":
            predictor = CoTrackerPredictor(checkpoint=None, window_len=16)
    if pretrained:
        if version == "2":
            state_dict = torch.hub.load_state_dict_from_url(_COTRACKER_URL, map_location="cpu")
        elif version == "2.1":
            state_dict = torch.hub.load_state_dict_from_url(_COTRACKER2v1_URL, map_location="cpu")
        else:
            raise Exception("Provided version does not exist")
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


def cotracker2v1(*, pretrained: bool = True, **kwargs):
    """
    CoTracker2 with stride 4 and window length 16.
    """
    return _make_cotracker_predictor(pretrained=pretrained, online=False, version="2.1", **kwargs)


def cotracker2v1_online(*, pretrained: bool = True, **kwargs):
    """
    Online CoTracker2 with stride 4 and window length 16.
    """
    return _make_cotracker_predictor(pretrained=pretrained, online=True, version="2.1", **kwargs)