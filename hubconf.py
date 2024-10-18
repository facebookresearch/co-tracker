# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

_COTRACKER2_URL = (
    "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth"
)
_COTRACKER2v1_URL = (
    "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2v1.pth"
)
_COTRACKER3_SCALED_OFFLINE_URL = (
    "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
)
_COTRACKER3_SCALED_ONLINE_URL = (
    "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth"
)


def _make_cotracker_predictor(
    *, pretrained: bool = True, online=False, version="3", **kwargs
):
    if online:
        from cotracker.predictor import CoTrackerOnlinePredictor

        if version == "2":
            predictor = CoTrackerOnlinePredictor(checkpoint=None, window_len=8, v2=True)
        elif version == "2.1":
            predictor = CoTrackerOnlinePredictor(
                checkpoint=None, window_len=16, v2=True
            )
        elif version == "3":
            predictor = CoTrackerOnlinePredictor(
                checkpoint=None, window_len=16, v2=False
            )
    else:
        from cotracker.predictor import CoTrackerPredictor

        if version == "2":
            predictor = CoTrackerPredictor(checkpoint=None, window_len=8, v2=True)
        elif version == "2.1":
            predictor = CoTrackerPredictor(checkpoint=None, window_len=16, v2=True)
        elif version == "3":
            predictor = CoTrackerPredictor(checkpoint=None, window_len=60, v2=False)
    if pretrained:
        if version == "2":
            state_dict = torch.hub.load_state_dict_from_url(
                _COTRACKER2_URL, map_location="cpu"
            )
        elif version == "2.1":
            state_dict = torch.hub.load_state_dict_from_url(
                _COTRACKER2v1_URL, map_location="cpu"
            )
        elif version == "3":
            if online:
                state_dict = torch.hub.load_state_dict_from_url(
                    _COTRACKER3_SCALED_ONLINE_URL, map_location="cpu"
                )
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    _COTRACKER3_SCALED_OFFLINE_URL, map_location="cpu"
                )
        else:
            raise Exception("Provided version does not exist")
        predictor.model.load_state_dict(state_dict)
    return predictor


def cotracker2(*, pretrained: bool = True, **kwargs):
    """
    CoTracker2 with stride 4 and window length 8. Can track up to 265*265 points jointly.
    """
    return _make_cotracker_predictor(pretrained=pretrained, online=False, version="2", **kwargs)


def cotracker2_online(*, pretrained: bool = True, **kwargs):
    """
    Online CoTracker2 with stride 4 and window length 8. Can track up to 265*265 points jointly.
    """
    return _make_cotracker_predictor(pretrained=pretrained, online=True, version="2", **kwargs)


def cotracker2v1(*, pretrained: bool = True, **kwargs):
    """
    CoTracker2 with stride 4 and window length 16.
    """
    return _make_cotracker_predictor(
        pretrained=pretrained, online=False, version="2.1", **kwargs
    )


def cotracker2v1_online(*, pretrained: bool = True, **kwargs):
    """
    Online CoTracker2 with stride 4 and window length 16.
    """
    return _make_cotracker_predictor(
        pretrained=pretrained, online=True, version="2.1", **kwargs
    )


def cotracker3_offline(*, pretrained: bool = True, **kwargs):
    """
    Scaled offline CoTracker3 with stride 4 and window length 16.
    """
    return _make_cotracker_predictor(
        pretrained=pretrained, online=False, version="3", **kwargs
    )


def cotracker3_online(*, pretrained: bool = True, **kwargs):
    """
    Scaled online CoTracker3 with stride 4 and window length 16.
    """
    return _make_cotracker_predictor(
        pretrained=pretrained, online=True, version="3", **kwargs
    )
