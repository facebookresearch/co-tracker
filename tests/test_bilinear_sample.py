# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import unittest

from cotracker.models.core.model_utils import bilinear_sampler


class TestBilinearSampler(unittest.TestCase):
    # Sample from an image (4d)
    def _test4d(self, align_corners):
        H, W = 4, 5
        # Construct a grid to obtain indentity sampling
        input = torch.randn(H * W).view(1, 1, H, W).float()
        coords = torch.meshgrid(torch.arange(H), torch.arange(W))
        coords = torch.stack(coords[::-1], dim=-1).float()[None]
        if not align_corners:
            coords = coords + 0.5
        sampled_input = bilinear_sampler(input, coords, align_corners=align_corners)
        torch.testing.assert_close(input, sampled_input)

    # Sample from a video (5d)
    def _test5d(self, align_corners):
        T, H, W = 3, 4, 5
        # Construct a grid to obtain indentity sampling
        input = torch.randn(H * W).view(1, 1, H, W).float()
        input = torch.stack([input, input + 1, input + 2], dim=2)
        coords = torch.meshgrid(torch.arange(T), torch.arange(W), torch.arange(H))
        coords = torch.stack(coords, dim=-1).float().permute(0, 2, 1, 3)[None]

        if not align_corners:
            coords = coords + 0.5
        sampled_input = bilinear_sampler(input, coords, align_corners=align_corners)
        torch.testing.assert_close(input, sampled_input)

    def test4d(self):
        self._test4d(align_corners=True)
        self._test4d(align_corners=False)

    def test5d(self):
        self._test5d(align_corners=True)
        self._test5d(align_corners=False)


# run the test
unittest.main()
