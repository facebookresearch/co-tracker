# coding: utf-8

from copy import deepcopy

from supervisely.sly_logger import logger
from supervisely.annotation.tag_meta import TagValueType
from supervisely.metric.metric_base import MetricsBase
from supervisely.metric.common import log_line, safe_ratio, sum_counters, TRUE_POSITIVE, TRUE_NEGATIVE, \
    FALSE_POSITIVE, FALSE_NEGATIVE, ACCURACY, PRECISION, RECALL, F1_MEASURE


RAW_COUNTERS = [TRUE_POSITIVE, TRUE_NEGATIVE, FALSE_POSITIVE, FALSE_NEGATIVE]


class ClassificationMetrics(MetricsBase):
    def __init__(self, tags_mapping, confidence_threshold=0):
        if len(tags_mapping) < 1:
            raise RuntimeError('At least one tags pair should be defined!')
        self._tags_mapping = tags_mapping.copy()
        self._confidence_threshold = confidence_threshold
        self._counters = {tag_name_gt: {counter: 0 for counter in RAW_COUNTERS} for tag_name_gt in
                          self._tags_mapping.keys()}

    def _classification_metrics(self, ann_1, ann_2):

        def is_passes_confidence_threshold(tag):
            if tag.meta.value_type == TagValueType.NONE:
                return True
            elif tag.meta.value_type == TagValueType.ANY_NUMBER:
                return tag.value >= self._confidence_threshold
            elif tag.meta.value_type == TagValueType.ANY_STRING or tag.meta.value_type == TagValueType.ONEOF_STRING:
                logger.warning("Classification tag '{}'".format(tag.name))
                return True

        current_metric_res = {}
        for tag_name_gt, tag_name_pred in self._tags_mapping.items():
            tag1 = ann_1.img_tags.get(tag_name_gt)
            tag2 = ann_2.img_tags.get(tag_name_pred)

            c1 = is_passes_confidence_threshold(tag1) if tag1 is not None else False
            c2 = is_passes_confidence_threshold(tag2) if tag2 is not None else False

            current_metric_res[tag_name_gt] = {
                TRUE_POSITIVE: int(c1 and c2),
                TRUE_NEGATIVE: int(not c1 and not c2),
                FALSE_POSITIVE: int(not c1 and c2),
                FALSE_NEGATIVE: int(c1 and not c2)
            }
        return current_metric_res

    def add_pair(self, ann_gt, ann_pred):
        res = self._classification_metrics(ann_gt, ann_pred)
        for tag_name_gt, met_data in res.items():
            for metric_name, metric_value in met_data.items():
                self._counters[tag_name_gt][metric_name] += metric_value

    @staticmethod
    def _calculate_complex_metrics(values):
        tp = values[TRUE_POSITIVE]
        tn = values[TRUE_NEGATIVE]
        fp = values[FALSE_POSITIVE]
        fn = values[FALSE_NEGATIVE]

        values[ACCURACY] = safe_ratio(tp + tn, tp + tn + fp + fn)
        values[PRECISION] = safe_ratio(tp, tp + fp)
        values[RECALL] = safe_ratio(tp, tp + fn)
        values[F1_MEASURE] = safe_ratio(2.0 * tp, 2.0 * tp + fp + fn)

    def get_metrics(self):
        result = deepcopy(self._counters)
        for pair_counters in result.values():
            self._calculate_complex_metrics(pair_counters)
        return result

    def get_total_metrics(self):
        result = sum_counters(self._counters.values(), (TRUE_POSITIVE, TRUE_NEGATIVE, FALSE_POSITIVE, FALSE_NEGATIVE))
        self._calculate_complex_metrics(result)
        return result

    def log_total_metrics(self):
        common_info = """
                P = condition positive (the number of real positive cases in the data)
                N = condition negative (the number of real negative cases in the data)
                TP = True Positive prediction
                TN = True Negative prediction
                FP = False Positive prediction (Type I error)
                FN = False Negative prediction (Type II error)
                Accuracy = (TP + TN)/(TP + TN + FP + FN) = TRUE/TOTAL
                Precision = TP / (TP + FP)
                Recall = TP / (TP + FN)
                F1-Measure = (2 * TP) / (2 * TP + FP + FN)
                """

        log_line()
        log_line(c='*')
        for line in common_info.split('\n'):
            line = line.strip()
            if len(line) > 0:
                logger.info(line.ljust(80))

        log_line(c='*')
        log_line()

        def print_evaluation_values(tag_pair_metrics):
            labels = [ACCURACY, PRECISION, RECALL, F1_MEASURE, TRUE_POSITIVE, TRUE_NEGATIVE, FALSE_POSITIVE,
                      FALSE_NEGATIVE]
            for label in labels:
                logger.info('    {0}:   {1:2.4f}'.format(label.ljust(16), tag_pair_metrics[label]))

        for i, (tag_name_gt, tag_metrics) in enumerate(self.get_metrics().items(), start=1):
            logger.info('{}) {} <--> {}:'.format(i, tag_name_gt, self._tags_mapping[tag_name_gt]))
            print_evaluation_values(tag_metrics)
            log_line()

        logger.info('Total values:')
        total_values = self.get_total_metrics()
        print_evaluation_values(total_values)
        log_line()

        log_line(c='*')

