# coding: utf-8

from collections import namedtuple

from supervisely.sly_logger import logger
from supervisely.metric.common import log_head, log_line, TOTAL_GROUND_TRUTH
from supervisely.metric.matching import filter_labels_by_name, match_labels_by_iou
from supervisely.metric.metric_base import MetricsBase

import numpy as np

MATCHES = 'matches'
AP = 'average-precision'

MatchWithConfidence = namedtuple('MatchWithConfidence', ['is_correct', 'confidence'])


class MAPMetric(MetricsBase):

    def __init__(self, class_mapping, iou_threshold, confidence_tag_name='confidence', confidence_threshold=0.0):
        if len(class_mapping) < 1:
            raise RuntimeError('At least one classes pair should be defined!')
        self._gt_to_pred_class_mapping = class_mapping.copy()
        self._pred_to_gt_class_mapping = {
            pred_name: gt_name for gt_name, pred_name in self._gt_to_pred_class_mapping.items()}
        if len(self._pred_to_gt_class_mapping) != len(self._gt_to_pred_class_mapping):
            raise ValueError('Mapping from ground truth to prediction class name is not 1-to-1: multiple ground truth '
                             'classes are mapped to the same prediction class. This is not supported for mAP '
                             'calculations.')

        self._iou_threshold = iou_threshold
        self._confidence_tag_name = confidence_tag_name
        self._confidence_threshold = confidence_threshold
        self._counters = {
            gt_cls: {MATCHES: [], TOTAL_GROUND_TRUTH: 0} for gt_cls in self._gt_to_pred_class_mapping.keys()}

    def _get_confidence_value(self, label):
        confidence_tag = label.tags.get(self._confidence_tag_name, None)
        return confidence_tag.value if confidence_tag is not None else None

    def add_pair(self, ann_gt, ann_pred):
        labels_gt = filter_labels_by_name(ann_gt.labels, self._gt_to_pred_class_mapping)
        all_labels_pred = [label for label in filter_labels_by_name(ann_pred.labels, self._pred_to_gt_class_mapping)]
        labels_pred = []
        for label in all_labels_pred:
            label_confidence = self._get_confidence_value(label)
            if label_confidence is None:
                logger.warn(f'Found a label with class {label.obj_class.name!r} that does not have a '
                            f'{self._confidence_tag_name!r} tag attached. Skipping this object for metric computation.')
            elif label_confidence >= self._confidence_threshold:
                labels_pred.append(label)
        match_result = match_labels_by_iou(labels_1=labels_gt, labels_2=labels_pred, img_size=ann_gt.img_size,
                                           iou_threshold=self._iou_threshold)
        for match in match_result.matches:
            gt_class = match.label_1.obj_class.name
            label_pred = match.label_2
            self._counters[gt_class][MATCHES].append(
                MatchWithConfidence(is_correct=(label_pred.obj_class.name == self._gt_to_pred_class_mapping[gt_class]),
                                    confidence=self._get_confidence_value(label_pred)))
        # Add unmatched predictions to the list as false positive matches.
        for umatched_pred in match_result.unmatched_labels_2:
            gt_class = self._pred_to_gt_class_mapping[umatched_pred.obj_class.name]
            self._counters[gt_class][MATCHES].append(
                MatchWithConfidence(is_correct=False, confidence=self._get_confidence_value(umatched_pred)))

        for label_1 in labels_gt:
            self._counters[label_1.obj_class.name][TOTAL_GROUND_TRUTH] += 1

    @staticmethod
    def _calculate_average_precision(gt_class, pred_class, pair_counters):
        if len(pair_counters[MATCHES]) == 0 or all(match.is_correct == False for match in pair_counters[MATCHES]):
            logger.warning('No matching samples for pair {!r} <-> {!r} have been detected. '
                           'MAP value for this pair will be set to 0.'.format(gt_class, pred_class))
            return 0

        sorted_matches = sorted(pair_counters[MATCHES], key=lambda match: match.confidence, reverse=True)
        correct_indicators = [int(match.is_correct) for match in sorted_matches]
        total_correct = np.cumsum(correct_indicators)
        recalls = total_correct / pair_counters[TOTAL_GROUND_TRUTH]
        precisions = total_correct / (np.arange(len(correct_indicators)) + 1)
        anchor_precisions = []
        for anchor_recall in np.linspace(0, 1, 11):
            points_above_recall = (recalls >= anchor_recall)
            anchor_precisions.append(np.max(precisions[points_above_recall]) if np.any(points_above_recall) else 0)
        return np.mean(anchor_precisions)

    def get_metrics(self):  # Macro-evaluation
        result = {gt_class:
                      {AP: self._calculate_average_precision(gt_class, self._gt_to_pred_class_mapping[gt_class],
                                                             pair_counters)}
                  for gt_class, pair_counters in self._counters.items()}
        return result

    @staticmethod
    def average_per_class_avg_precision(per_class_metrics):
        return np.mean([class_metrics[AP] for class_metrics in per_class_metrics.values()])

    def get_total_metrics(self):
        return {AP: self.average_per_class_avg_precision(self.get_metrics())}

    def log_total_metrics(self):
        log_line()
        log_head(' Result metrics values for {} IoU threshold '.format(self._iou_threshold))

        classes_values = self.get_metrics()
        for i, (cls_gt, pair_values) in enumerate(classes_values.items()):
            average_precision = pair_values[AP]
            log_line()
            log_head(' Results for pair of classes <<{} <-> {}>>  '.format(cls_gt,
                                                                           self._gt_to_pred_class_mapping[cls_gt]))
            logger.info('Average Precision (AP): {}'.format(average_precision))

        log_line()
        log_head(' Mean metrics values ')
        logger.info('Mean Average Precision (mAP): {}'.format(self.get_total_metrics()[AP]))
        log_line()
