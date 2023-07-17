# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from cotracker.models.core.model_utils import reduce_masked_mean

EPS = 1e-6


def balanced_ce_loss(pred, gt, valid=None):
    total_balanced_loss = 0.0
    for j in range(len(gt)):
        B, S, N = gt[j].shape
        # pred and gt are the same shape
        for (a, b) in zip(pred[j].size(), gt[j].size()):
            assert a == b  # some shape mismatch!
        # if valid is not None:
        for (a, b) in zip(pred[j].size(), valid[j].size()):
            assert a == b  # some shape mismatch!

        pos = (gt[j] > 0.95).float()
        neg = (gt[j] < 0.05).float()

        label = pos * 2.0 - 1.0
        a = -label * pred[j]
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

        pos_loss = reduce_masked_mean(loss, pos * valid[j])
        neg_loss = reduce_masked_mean(loss, neg * valid[j])

        balanced_loss = pos_loss + neg_loss
        total_balanced_loss += balanced_loss / float(N)
    return total_balanced_loss


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        assert D == 2
        B, S1, N = vis[j].shape
        B, S2, N = valids[j].shape
        assert S == S1
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            i_loss = (flow_pred - flow_gt[j]).abs()  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            flow_loss += i_weight * reduce_masked_mean(i_loss, valids[j])
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss / float(N)
    return total_flow_loss
