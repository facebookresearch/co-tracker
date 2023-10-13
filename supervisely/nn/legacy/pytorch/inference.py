# coding: utf-8
import numpy as np

import torch
import torch.nn.functional as torch_functional

from supervisely.imaging import image as sly_image
from supervisely.nn.pytorch.cuda import cuda_variable


def infer_per_pixel_scores_single_image(model, raw_input, out_shape, apply_softmax=True):
    """
    Performs inference with PyTorch model and resize predictions to a given size.

    Args:
        model: PyTorch model inherited from torch.Module class.
        raw_input: PyTorch Tensor
        out_shape: Output size (height, width).
        apply_softmax: Whether to apply softmax function after inference or not.
    Returns:
        Inference resulting numpy array resized to a given size.
    """
    model_input = torch.stack([raw_input], 0)  # add dim #0 (batch size 1)
    model_input = cuda_variable(model_input, volatile=True)

    output = model(model_input)
    if apply_softmax:
        output = torch_functional.softmax(output, dim=1)
    output = output.data.cpu().numpy()[0]  # from batch to 3d

    pred = np.transpose(output, (1, 2, 0))
    return sly_image.resize(pred, out_shape)
