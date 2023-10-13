# coding: utf-8
import json

from supervisely import Bitmap, logger, ObjClassCollection, ObjClass
from supervisely.io import fs


BACKGROUND = 'background'
NEUTRAL = 'neutral'

CONTINUE_TRAINING = 'continue_training'
TRANSFER_LEARNING = 'transfer_learning'


def make_out_classes(in_project_classes, geometry_type, exclude_titles=None, extra_classes=None):
    exclude_titles = exclude_titles or []
    extra_classes = extra_classes or dict()

    if any(title in extra_classes for title in exclude_titles):
        raise RuntimeError(
            'Unable to construct class name to integer id map: extra classes names overlap with excluded classes.')

    out_classes = ObjClassCollection()
    for in_class in in_project_classes:
        if in_class.name not in exclude_titles:
            out_classes = out_classes.add(ObjClass(in_class.name, geometry_type, in_class.color))

    for title, color in extra_classes.items():
        if not out_classes.has_key(title):
            out_classes = out_classes.add(ObjClass(title, geometry_type, color))
    return out_classes


def create_segmentation_classes(in_project_classes, special_classes_config, bkg_input_idx,
                                weights_init_type, model_config_fpath, class_to_idx_config_key, start_class_id=1):
    extra_classes = {}
    special_class_ids = {}
    bkg_title = special_classes_config.get(BACKGROUND, None)
    if bkg_title is not None:
        extra_classes = {bkg_title: [34, 34, 34]} # Default background color
        special_class_ids[bkg_title] = bkg_input_idx

    exclude_titles = []
    neutral_title = special_classes_config.get(NEUTRAL, None)
    if neutral_title is not None:
        exclude_titles.append(neutral_title)
    out_classes = make_out_classes(in_project_classes, geometry_type=Bitmap, exclude_titles=exclude_titles,
                                   extra_classes=extra_classes)

    logger.info('Determined model out classes', extra={'out_classes': list(out_classes)})

    in_project_class_to_idx = make_new_class_to_idx_map(in_project_classes, start_class_id=start_class_id,
                                                        preset_class_ids=special_class_ids,
                                                        exclude_titles=exclude_titles)
    class_title_to_idx = infer_training_class_to_idx_map(weights_init_type,
                                                         in_project_class_to_idx,
                                                         model_config_fpath,
                                                         class_to_idx_config_key,
                                                         special_class_ids=special_class_ids)
    logger.info('Determined class mapping.', extra={'class_mapping': class_title_to_idx})
    return out_classes, class_title_to_idx


def make_new_class_to_idx_map(in_project_classes: ObjClassCollection, start_class_id: int,
                              preset_class_ids: dict = None, exclude_titles: list = None):
    preset_class_ids = preset_class_ids or dict()
    exclude_titles = exclude_titles or []
    if any(title in preset_class_ids for title in exclude_titles):
        raise RuntimeError(
            'Unable to construct class name to integer id map: preset classes names overlap with excluded classes.')
    sorted_titles = sorted([obj_class.name for obj_class in in_project_classes])
    if len(sorted_titles) != len(set(sorted_titles)):
        raise RuntimeError('Unable to construct class name to integer id map: class names are not unique.')

    class_title_to_idx = preset_class_ids.copy()
    next_title_id = start_class_id
    already_used_ids = set(class_title_to_idx.values())
    for title in sorted_titles:
        if title not in class_title_to_idx and title not in exclude_titles:
            while next_title_id in already_used_ids:
                next_title_id += 1
            class_title_to_idx[title] = next_title_id
            already_used_ids.add(next_title_id)
    return class_title_to_idx


def read_validate_model_class_to_idx_map(model_config_fpath, in_project_classes_set, class_to_idx_config_key):
    """Reads class id --> int index mapping from the model config; checks that the set of classes matches the input."""

    if not fs.file_exists(model_config_fpath):
        raise RuntimeError('Unable to continue_training, config for previous training wasn\'t found.')

    with open(model_config_fpath) as fin:
        model_config = json.load(fin)

    model_class_mapping = model_config.get(class_to_idx_config_key, None)
    if model_class_mapping is None:
        raise RuntimeError('Unable to continue_training, model does not have class mapping information.')
    model_classes_set = set(model_class_mapping.keys())

    if model_classes_set != in_project_classes_set:
        error_message_text = 'Unable to continue_training, sets of classes for model and dataset do not match.'
        logger.critical(
            error_message_text, extra={'model_classes': model_classes_set, 'dataset_classes': in_project_classes_set})
        raise RuntimeError(error_message_text)
    return model_class_mapping.copy()


def infer_training_class_to_idx_map(weights_init_type, in_project_class_to_idx, model_config_fpath,
                                    class_to_idx_config_key, special_class_ids=None):
    if weights_init_type == TRANSFER_LEARNING:
        logger.info('Transfer learning mode, using a class mapping created from scratch.')
        class_title_to_idx = in_project_class_to_idx
    elif weights_init_type == CONTINUE_TRAINING:
        logger.info('Continued training mode, reusing the existing class mapping from the model.')
        class_title_to_idx = read_validate_model_class_to_idx_map(
            model_config_fpath=model_config_fpath,
            in_project_classes_set=set(in_project_class_to_idx.keys()),
            class_to_idx_config_key=class_to_idx_config_key)
    else:
        raise RuntimeError('Unknown weights init type: {}'.format(weights_init_type))

    if special_class_ids is not None:
        for class_title, requested_class_id in special_class_ids.items():
            effective_class_id = class_title_to_idx[class_title]
            if requested_class_id != effective_class_id:
                error_msg = ('Unable to start training. Effective integer id for class {} does not match the ' +
                             'requested value in the training config ({} vs {}).'.format(
                                 class_title, effective_class_id, requested_class_id))
                logger.critical(error_msg, extra={'class_title_to_idx': class_title_to_idx,
                                                  'special_class_ids': special_class_ids})
                raise RuntimeError(error_msg)
    return class_title_to_idx
