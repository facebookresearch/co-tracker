# coding: utf-8

from supervisely.sly_logger import logger
from supervisely.annotation.annotation import Annotation
from supervisely.metric.common import log_head, log_line, CONFUSION_MATRIX, UNMATCHED_GT, UNMATCHED_PREDICTIONS
from supervisely.metric.matching import filter_labels_by_name, match_labels_by_iou
from supervisely.metric.metric_base import MetricsBase


class ConfusionMatrixMetric(MetricsBase):

    def __init__(self, class_mapping, iou_threshold):
        if len(class_mapping) < 1:
            raise RuntimeError('At least one classes pair should be defined!')
        self._class_mapping = class_mapping.copy()
        self._iou_threshold = iou_threshold
        self._confusion_matrix = {(cls_gt, cls_pred): 0
                                  for cls_gt in class_mapping.keys()
                                  for cls_pred in class_mapping.values()}
        self._unmatched_gt = {cls_gt: 0 for cls_gt in class_mapping.keys()}
        self._unmatched_pred = {cls_pred: 0 for cls_pred in class_mapping.values()}

    def add_pair(self, ann_gt: Annotation, ann_pred: Annotation):
        labels_gt = filter_labels_by_name(ann_gt.labels, self._unmatched_gt)
        labels_pred = filter_labels_by_name(ann_pred.labels, self._unmatched_pred)
        match_result = match_labels_by_iou(labels_1=labels_gt, labels_2=labels_pred, img_size=ann_gt.img_size,
                                           iou_threshold=self._iou_threshold)
        for match in match_result.matches:
            self._confusion_matrix[match.label_1.obj_class.name, match.label_2.obj_class.name] += 1
        for unmatched_gt_label in match_result.unmatched_labels_1:
            self._unmatched_gt[unmatched_gt_label.obj_class.name] += 1
        for unmatched_pred_label in match_result.unmatched_labels_2:
            self._unmatched_pred[unmatched_pred_label.obj_class.name] += 1

    def get_metrics(self):
        return {CONFUSION_MATRIX: self._confusion_matrix.copy(),
                UNMATCHED_GT: self._unmatched_gt.copy(),
                UNMATCHED_PREDICTIONS: self._unmatched_pred.copy()}

    def get_total_metrics(self):
        return self.get_metrics()

    def log_total_metrics(self):
        def exp_one(arg):
            return str(arg).center(20)

        def exp_arg(args):
            return [exp_one(arg) for arg in args]

        log_line()
        log_head(' Result metrics values for {} IoU threshold '.format(self._iou_threshold))
        log_head(' Confusion matrix ')

        sorted_gt_names = sorted(self._class_mapping.keys())
        pred_names = [self._class_mapping[gt_name] for gt_name in sorted_gt_names]
        logger.info(''.join(exp_arg([''] + pred_names + ['False Negatives'])))
        for gt_name in sorted_gt_names:
            logger.info(''.join([exp_one(gt_name)] +
                                exp_arg([self._confusion_matrix[gt_name, pred_name] for pred_name in pred_names]) +
                                [exp_one(self._unmatched_gt[gt_name])]))
            log_line()
        logger.info(''.join([exp_one('False Positives')] +
                            exp_arg([self._unmatched_pred[pred_name] for pred_name in pred_names]) +
                            [exp_one('0')]))
        log_line()
