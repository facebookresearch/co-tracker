# coding: utf-8
from copy import deepcopy

from supervisely.nn.hosted.constants import MODEL

# Copied from supervisely.nn.hosted.inference_modes
# TODO factor out to a separate common set of constants
MODEL_CLASSES = 'model_classes'
MODEL_SETTINGS = 'model_settings'
NAME = 'name'

MODE = 'mode'
SOURCE = 'source'


def maybe_convert_from_v1_inference_task_config(raw_config):
    if MODEL in raw_config:
        # If the config has sections with new (not v1) names, do not try to convert - most likely the config is
        # already in the new format.
        return raw_config
    else:
        config = {MODE: deepcopy(raw_config.get(MODE, {}))}
        if SOURCE in config[MODE]:
            config[MODE][NAME] = config[MODE].pop(SOURCE)
        if MODEL_CLASSES not in config[MODE]:
            config[MODE][MODEL_CLASSES] = deepcopy(raw_config.get(MODEL_CLASSES, {}))
        config[MODEL] = {}
        for k, v in raw_config.items():
            if k not in (MODE, MODEL_CLASSES):
                config[MODEL][k] = deepcopy(v)
        return config


def maybe_convert_from_deploy_task_config(raw_config):
    if 'connection' in raw_config and MODEL_SETTINGS in raw_config:
        raw_config[MODEL] = raw_config[MODEL_SETTINGS]
    return raw_config

