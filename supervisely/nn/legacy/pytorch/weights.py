# coding: utf-8
import os.path

import torch
from torch.nn import Parameter


class WeightsRW:
    """
    Help to load weights for PyTorch model from file and save current model state dict into given place.

    Args:
        model_dir: Path to the folder for storing weights.
        model_file: Name of weights file(default: model.pt).
    """
    def __init__(self, model_dir, model_file=None):
        self._weights_fpath = os.path.join(model_dir, model_file or 'model.pt')

    def save(self, model):
        torch.save(model.state_dict(), self._weights_fpath)

    @staticmethod
    def _transfer_params(src_state, dest_model):
        """Copies parameters and buffers from :attr:`src_state` into
        :attr:`dest_model` module and its descendants.

        Arguments:
            src_state (dict): A dict containing parameters and
                persistent buffers.
            dest_model: model
        """

        dest_state = dest_model.state_dict()
        for name, param in src_state.items():
            if name in dest_state:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    dest_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, dest_state[name].size(), param.size()))

    def load_for_transfer_learning(self, model, ignore_matching_layers=None, logger=None):
        ignore_matching_layers = ignore_matching_layers or []
        snapshot_weights = torch.load(self._weights_fpath)

        # Remove the layers matching the requested patterns (usually done for the head layers in transfer learning).
        # Make an explicit set for easier logging.
        to_delete = set(el for el in snapshot_weights.keys() if
                        any(delete_substring in el for delete_substring in ignore_matching_layers))
        snapshot_weights = {k: v for k, v in snapshot_weights.items() if k not in to_delete}
        if len(to_delete) > 0 and logger is not None:
            logger.info('Skip weight init for output layers.', extra={'layer_names': sorted(to_delete)})

        self._transfer_params(snapshot_weights, model)
        return model

    def load_strictly(self, model, logger=None):
        snapshot_weights = torch.load(self._weights_fpath)

        # Make sure the sets of parameters exactly match between the model and the snapshot.
        snapshot_keys = set(snapshot_weights.keys())
        model_keys = set(model.state_dict().keys())
        snapshot_extra_keys = snapshot_keys - model_keys
        if len(snapshot_extra_keys) > 0:
            raise KeyError('Parameters found in the snapshot file, but not in the model: {}'.format(snapshot_extra_keys))
        model_extra_keys = model_keys - snapshot_keys
        if len(model_extra_keys) > 0 and logger is not None:
            logger.warning('Model parameters missing from the snapshot file: {}'.format(model_extra_keys))

        self._transfer_params(snapshot_weights, model)
        return model
