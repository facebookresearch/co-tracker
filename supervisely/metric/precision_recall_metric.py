# coding: utf-8

from copy import deepcopy

from supervisely.sly_logger import logger
from supervisely.metric.matching import filter_labels_by_name, match_labels_by_iou
from supervisely.metric.metric_base import MetricsBase
from supervisely.metric.common import log_head, log_line, safe_ratio, sum_counters, TRUE_POSITIVE, PRECISION, \
    RECALL, TOTAL_GROUND_TRUTH, TOTAL_PREDICTIONS

RAW_COUNTERS = [TRUE_POSITIVE, TOTAL_GROUND_TRUTH, TOTAL_PREDICTIONS]


class PrecisionRecallMetric(MetricsBase):

    def __init__(self, class_mapping, iou_threshold):
        if len(class_mapping) < 1:
            raise RuntimeError('At least one classes pair should be defined!')
        self._gt_to_pred_class_mapping = class_mapping.copy()
        self._pred_to_gt_class_mapping = {v: k for k, v in class_mapping.items()}
        self._iou_threshold = iou_threshold
        self._counters = {gt_cls: {counter: 0 for counter in RAW_COUNTERS}
                          for gt_cls in self._gt_to_pred_class_mapping.keys()}

    def add_pair(self, ann_gt, ann_pred):
        for key in self._gt_to_pred_class_mapping.keys():
            labels_gt = filter_labels_by_name(ann_gt.labels, [key])
            labels_pred = filter_labels_by_name(ann_pred.labels, [self._gt_to_pred_class_mapping[key]])
            match_result = match_labels_by_iou(labels_1=labels_gt, labels_2=labels_pred, img_size=ann_gt.img_size,
                                               iou_threshold=self._iou_threshold)
            # TODO unify with confusion matrix ?
            for match in match_result.matches:
                self._counters[match.label_1.obj_class.name][TRUE_POSITIVE] += 1
            for label_gt in labels_gt:
                self._counters[label_gt.obj_class.name][TOTAL_GROUND_TRUTH] += 1
            for label_pred in labels_pred:
                self._counters[self._pred_to_gt_class_mapping[label_pred.obj_class.name]][TOTAL_PREDICTIONS] += 1

    @staticmethod
    def _compute_composite_metrics(metrics_dict):
        metrics_dict[PRECISION] = safe_ratio(metrics_dict[TRUE_POSITIVE], metrics_dict[TOTAL_PREDICTIONS])
        metrics_dict[RECALL] = safe_ratio(metrics_dict[TRUE_POSITIVE], metrics_dict[TOTAL_GROUND_TRUTH])

    def get_metrics(self):
        result = deepcopy(self._counters)
        for pair_counters in result.values():
            self._compute_composite_metrics(pair_counters)
        return result

    def get_total_metrics(self):
        result = sum_counters(self._counters.values(), RAW_COUNTERS)
        self._compute_composite_metrics(result)
        return result

    def log_total_metrics(self):
        log_line()
        log_head(' Result metrics values for {} IoU threshold '.format(self._iou_threshold))

        for i, (gt_class, values) in enumerate(self.get_metrics().items()):
            log_line()
            log_head(' Results for pair of classes <<{} <-> {}>>  '.format(gt_class,
                                                                           self._gt_to_pred_class_mapping[gt_class]))
            logger.info('Precision: {}'.format(values[PRECISION]))
            logger.info('Recall: {}'.format(values[RECALL]))

        log_line()
        log_head(' Total metrics values ')
        total_values = self.get_total_metrics()
        logger.info('Precision: {}'.format(total_values[PRECISION]))
        logger.info('Recall: {}'.format(total_values[RECALL]))
        log_line()
