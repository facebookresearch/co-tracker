# coding: utf-8

import numpy as np

from supervisely.sly_logger import logger
from supervisely.metric.metric_base import MetricsBase
from supervisely.metric.common import render_labels_for_class_name, safe_ratio, sum_counters, TOTAL_GROUND_TRUTH


INTERSECTION = 'intersection'
UNION = 'union'
IOU = 'iou'


def get_intersection(mask_1, mask_2):
    return (mask_1 & mask_2).sum()


def get_union(mask_1, mask_2):
    return (mask_1 | mask_2).sum()


def get_iou(mask_1, mask_2):
    return safe_ratio(get_intersection(mask_1, mask_2), get_union(mask_1, mask_2))


def _iou_log_line(iou, intersection, union):
    return 'IoU = {:.6f},  mean intersection = {:.6f}, mean union = {:.6f}'.format(iou, intersection, union)

def render_labels_as_binary_mask(labels, class_title, mask):
    for label in ann.labels:
        if label.obj_class.name == class_title:
                label.geometry.draw(mask, True)

class IoUMetric(MetricsBase):

    def __init__(self, class_mapping):
        self._class_mapping = class_mapping.copy()
        if len(self._class_mapping) < 1:
            raise RuntimeError('At least one classes pair should be defined.')
        self._counters = {cls_gt: {INTERSECTION: 0, UNION: 0, TOTAL_GROUND_TRUTH: 0}
                          for cls_gt in self._class_mapping.keys()}

    def add_pair(self, ann_gt, ann_pred):
        img_size = ann_gt.img_size
        for cls_gt, cls_pred in self._class_mapping.items():
            mask_gt, mask_pred = np.full(img_size, False), np.full(img_size, False)
            render_labels_as_binary_mask(ann_gt.labels, cls_gt, mask_gt)
            render_labels_as_binary_mask(ann_pred.labels, cls_pred, mask_pred)
            class_pair_counters = self._counters[cls_gt]
            class_pair_counters[INTERSECTION] += get_intersection(mask_gt, mask_pred)
            class_pair_counters[UNION] += get_union(mask_gt, mask_pred)
            class_pair_counters[TOTAL_GROUND_TRUTH] += 1

    def get_metrics(self):
        return {cls_gt: {INTERSECTION: safe_ratio(class_counters[INTERSECTION], class_counters[TOTAL_GROUND_TRUTH]),
                         UNION: safe_ratio(class_counters[UNION], class_counters[TOTAL_GROUND_TRUTH]),
                         IOU: safe_ratio(class_counters[INTERSECTION], class_counters[UNION])}
                for cls_gt, class_counters in self._counters.items()}

    def get_total_metrics(self):
        result = sum_counters(self._counters.values(), (INTERSECTION, UNION))
        result[IOU] = safe_ratio(result[INTERSECTION], result[UNION])
        return result

    def log_total_metrics(self):
        logger.info('**************** Result IoU metric values ****************')
        logger.info('NOTE! Values for "intersection" and "union" are in pixels.')
        for i, (cls_gt, values) in enumerate(self.get_metrics().items(), start=1):
            iou_line = _iou_log_line(values[IOU], values[INTERSECTION], values[UNION])
            logger.info('{}. Classes {} <-> {}:   {}'.format(i, cls_gt, self._class_mapping[cls_gt], iou_line))

        total_values = self.get_total_metrics()
        logger.info(
            'Total:   {}'.format(_iou_log_line(total_values[IOU], total_values[INTERSECTION], total_values[UNION])))
