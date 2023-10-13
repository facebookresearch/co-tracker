# coding: utf-8

import numpy as np

from supervisely.metric.metric_base import MetricsBase
from supervisely.metric.common import render_labels_for_classes, safe_ratio, TRUE_POSITIVE, TRUE_NEGATIVE, \
    FALSE_POSITIVE, FALSE_NEGATIVE, PRECISION, RECALL, ACCURACY, TOTAL


class PixelAccuracyMetric(MetricsBase):
    def __init__(self, class_mapping):
        self._class_mapping = class_mapping.copy()
        if len(self._class_mapping) < 1:
            raise RuntimeError('At least one classes pair should be defined.')
        self._counters = {cls_gt: {TRUE_POSITIVE: 0, TRUE_NEGATIVE: 0, FALSE_POSITIVE: 0, FALSE_NEGATIVE: 0, TOTAL: 0}
                          for cls_gt in self._class_mapping.keys()}
        self._total_counters = {TRUE_POSITIVE: 0, TRUE_NEGATIVE: 0, FALSE_POSITIVE: 0, FALSE_NEGATIVE: 0, TOTAL: 0}
        self._gt_colors = {}
        self._pred_colors = {}
        for class_idx, (gt_class, pred_class) in enumerate(self._class_mapping.items(), start=1):
            self._gt_colors[gt_class] = class_idx
            self._pred_colors[pred_class] = class_idx

    def add_pair(self, ann_gt, ann_pred):
        img_size = ann_gt.img_size
        render_gt, render_pred = np.zeros(img_size, dtype=np.int32), np.zeros(img_size, dtype=np.int32)
        render_labels_for_classes(ann_gt.labels, self._gt_colors, render_gt, missing_classes_color=0)
        render_labels_for_classes(ann_pred.labels, self._pred_colors, render_pred, missing_classes_color=0)

        for cls_gt, cls_pred in self._class_mapping.items():
            mask_gt = (render_gt == self._gt_colors[cls_gt])
            mask_pred = (render_pred == self._pred_colors[cls_pred])
            class_pair_counters = self._counters[cls_gt]
            class_pair_counters[TRUE_POSITIVE] += (mask_gt & mask_pred).sum()
            class_pair_counters[TRUE_NEGATIVE] += (np.logical_not(mask_gt) & np.logical_not(mask_pred)).sum()
            class_pair_counters[FALSE_POSITIVE] += (np.logical_not(mask_gt) & mask_pred).sum()
            class_pair_counters[FALSE_NEGATIVE] += (mask_gt & np.logical_not(mask_pred)).sum()
            class_pair_counters[TOTAL] += mask_gt.size

        self._total_counters[TRUE_POSITIVE] += ((render_gt == render_pred) & (render_pred != 0)).sum()
        self._total_counters[TRUE_NEGATIVE] += ((render_gt == 0) & (render_pred == 0)).sum()
        self._total_counters[FALSE_POSITIVE] += ((render_gt != render_pred) & (render_pred != 0)).sum()
        self._total_counters[FALSE_NEGATIVE] += ((render_gt != 0) & (render_pred == 0)).sum()
        self._total_counters[TOTAL] += render_gt.size

    @staticmethod
    def _metrics_from_counters(counters):
        return {PRECISION: safe_ratio(counters[TRUE_POSITIVE], counters[TRUE_POSITIVE] + counters[FALSE_POSITIVE]),
                RECALL: safe_ratio(counters[TRUE_POSITIVE], counters[TRUE_POSITIVE] + counters[FALSE_NEGATIVE]),
                ACCURACY: safe_ratio(counters[TRUE_POSITIVE] + counters[TRUE_NEGATIVE], counters[TOTAL])}

    def get_metrics(self):
        return {cls_gt: self._metrics_from_counters(class_counters) for cls_gt, class_counters in
                self._counters.items()}

    def get_total_metrics(self):
        return self._metrics_from_counters(self._total_counters)
