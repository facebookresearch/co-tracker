# coding: utf-8
import math


VAL_EVERY = 'val_every'


class EvalPlanner:
    """
    Helps determine whether to start validation in current train step or not.

    Args:
        epochs: Number of train epochs.
        val_every:  Validation period by epoch (value 0.5 mean 2 validations per epoch).
    """
    def __init__(self, epochs, val_every):
        self.epochs = epochs
        self.val_every = val_every
        self.total_val_cnt = self.validations_cnt(epochs, val_every)
        self._val_cnt = 0

    @property
    def performed_val_cnt(self):
        """int: Number of performed validations."""
        return self._val_cnt

    @staticmethod
    def validations_cnt(ep_float, val_every):
        res = math.floor(ep_float / val_every + 1e-9)
        return res

    def validation_performed(self):
        """Increments number of performed validations"""
        self._val_cnt += 1

    def need_validation(self, epoch_flt):
        """
        Determines whether to start validation or not.

        Args:
            epoch_flt: Current train step.
        Returns:
            True if validation is needed in current ste, False otherwise.
        """
        req_val_cnt = self.validations_cnt(epoch_flt, self.val_every)
        need_val = req_val_cnt > self._val_cnt
        return need_val
