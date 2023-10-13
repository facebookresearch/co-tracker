# coding: utf-8
import numpy as np


class MultiClassAccuracy:
    """
    Compute multi class accuracy for image segmentation task.

    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the accuracy value.
        squeeze_targets: Whether to squeeze targets array along 1st dimension.

    """
    def __init__(self, ignore_index: int=None, squeeze_targets: bool=True):
        self._ignore_index = ignore_index
        self._squeeze_targets = squeeze_targets

    def __call__(self, outputs, targets):
        outputs = outputs.data.cpu().numpy()
        outputs = np.argmax(outputs, axis=1)

        targets = targets.data.cpu().numpy()
        if self._squeeze_targets:
            targets = np.squeeze(targets, 1)  # 4d to 3d

        total_pixels = np.prod(outputs.shape)
        if self._ignore_index is not None:
            total_pixels -= np.sum((targets == self._ignore_index).astype(int))
        correct_pixels = np.sum((outputs == targets).astype(int))
        res = correct_pixels / total_pixels
        return res
