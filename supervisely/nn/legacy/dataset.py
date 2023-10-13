# coding: utf-8
import random
from collections import defaultdict
from threading import Lock

import numpy as np

from supervisely import logger
from supervisely.annotation.annotation import Annotation
from supervisely.imaging import image as sly_image


def samples_by_tags(required_tags, project):
    """
    Split samples from project by tags
    :param required_tags: list of tags names
    :param project: supervisely `Project` class object
    :return:
    """
    img_annotations_groups = defaultdict(list)
    for dataset in project:
        for item_name in dataset:
            item_paths = dataset.get_item_paths(item_name)
            ann = Annotation.load_json_file(path=item_paths.ann_path, project_meta=project.meta)
            img_tags = ann.img_tags
            for required_tag in required_tags:
                if img_tags.has_key(required_tag):
                    # TODO migrate to ItemPath objects for img_annotations_groups
                    img_annotations_groups[required_tag].append((item_paths.img_path, item_paths.ann_path))
    return img_annotations_groups


def ensure_samples_nonempty(samples, tag_name, project_meta):
    """

    Args:
        samples: list of pairs (image path, annotation path).
        tag_name: tag name for messages.
        project_meta: input project meta object.
    Returns: None

    """
    if len(samples) < 1:
        raise RuntimeError('There are no annotations with tag "{}"'.format(tag_name))

    for _, ann_path in samples:
        ann = Annotation.load_json_file(ann_path, project_meta)
        if len(ann.labels) > 0:
            return

    raise RuntimeError('There are no objects in annotations with tag "{}"'.format(tag_name))


class CorruptedSampleCatcher(object):
    def __init__(self, allow_corrupted_cnt):
        self.fails_allowed = allow_corrupted_cnt
        self._failed_uids = set()
        self._lock = Lock()

    def exec(self, uid, log_dct, f, *args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            self._lock.acquire()
            if uid not in self._failed_uids:
                self._failed_uids.add(uid)
                logger.warn('Sample processing error.', exc_info=True, extra={**log_dct, 'exc_str': str(e)})
            fail_cnt = len(self._failed_uids)
            self._lock.release()

            if fail_cnt > self.fails_allowed:
                raise RuntimeError('Too many errors occurred while processing samples. '
                                   'Allowed: {}.'.format(self.fails_allowed))


class SlyDataset:
    """
    SlyDataset class is mostly compatible with PyTorch 'Dataset' class except the output image format is HWC, not CHW.

    Override _get_sample_impl to convert raw Numpy arrays to the proper tensor format for your model.
    """

    def __init__(self, project_meta, samples, out_size, class_mapping, bkg_color, allow_corrupted_cnt=0,
                 catcher_retries=100):
        self._project_meta = project_meta
        self._samples = samples
        self._out_size = tuple(out_size)
        self._class_mapping = class_mapping
        self._bkg_color = bkg_color
        self._sample_catcher = CorruptedSampleCatcher(allow_corrupted_cnt)
        self._catcher_retries = catcher_retries

    def __len__(self):
        return len(self._samples)

    def load_annotation(self, fpath):
        # will not resize figures: resize gt instead
        return Annotation.load_json_file(fpath, self._project_meta)

    def make_gt(self, image_shape, ann, ignore_not_mapped_classes=False):
        # int32 instead of int64 because opencv cannot draw on int64 bitmaps.
        gt = np.ones(image_shape[:2], dtype=np.int32) * self._bkg_color  # default bkg
        for label in ann.labels:
            gt_color = self._class_mapping.get(label.obj_class.name, None)
            if gt_color is None:
                # TODO: Factor out this check to lib
                if not ignore_not_mapped_classes:
                    raise RuntimeError('Missing class mapping (title to index). Class {}.'.format(label.obj_class.name))
            else:
                label.geometry.draw(gt, gt_color)
        gt = sly_image.resize_inter_nearest(gt, self._out_size).astype(np.int64)
        return gt

    def _get_sample_impl(self, img_fpath, ann_fpath):
        img = sly_image.read(img_fpath)
        ann = self.load_annotation(ann_fpath)
        gt = self.make_gt(img.shape, ann)
        img = sly_image.resize(img, self._out_size)
        return img, gt

    def __getitem__(self, index):
        for att in range(self._catcher_retries):
            img_path, ann_path = self._samples[index]
            res = self._sample_catcher.exec(index,
                                            {'img': img_path,
                                             'ann': ann_path},
                                            self._get_sample_impl,
                                            img_path,
                                            ann_path)
            if res is not None:
                return res
            index = random.randrange(len(self._samples))  # must be ok for large ds
        raise RuntimeError('Unable to load some correct sample.')


def partition_train_val(num_samples, val_fraction):
    """Returns a bool array indicating whether a given sample falls into a train fold (otherwise samples fall to val).
    """

    if num_samples <= 1:
        raise ValueError('Need at least 2 samples to prepare a training/validation split (at least 1 each for training '
                         'and validation).')

    val_boundary = round(num_samples * val_fraction)
    # Make sure there is at least one sample in both training and validation fold,
    # adjust boundary if necessary.
    if val_boundary <= 0:
        val_boundary = 1
    elif val_boundary >= num_samples:
        val_boundary = num_samples - 1
    is_train_sample = ([False] * val_boundary) + ([True] * (num_samples - val_boundary))
    random.shuffle(is_train_sample)
    return is_train_sample
