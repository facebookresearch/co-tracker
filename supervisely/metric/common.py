# coding: utf-8

from supervisely.project.project import Project
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.annotation.tag_meta import TagValueType
from supervisely.sly_logger import logger


CLASSES_MAPPING = 'classes_mapping'
TAGS_MAPPING = 'tags_mapping'
CONFIDENCE_THRESHOLD = 'confidence_threshold'

TRUE_POSITIVE = 'true-positive'
TRUE_NEGATIVE = 'true-negative'
FALSE_POSITIVE = 'false-positive'
FALSE_NEGATIVE = 'false-negative'

ACCURACY = 'accuracy'
PRECISION = 'precision'
RECALL = 'recall'
F1_MEASURE = 'F1-measure'
TOTAL = 'total'
TOTAL_GROUND_TRUTH = 'total-ground-truth'
TOTAL_PREDICTIONS = 'total-predictions'

CONFUSION_MATRIX = 'confusion-matrix'
UNMATCHED_GT = 'unmatched-gt'
UNMATCHED_PREDICTIONS = 'unmatched-predictions'


def check_class_mapping(first_project: Project, second_project: Project, classes_mapping: dict) -> None:
    for k, v in classes_mapping.items():
        if first_project.meta.obj_classes.get(k) is None:
            raise RuntimeError('Class {} does not exist in input project "{}".'.format(k, first_project.name))
        if second_project.meta.obj_classes.get(v) is None:
            raise RuntimeError('Class {} does not exist in input project "{}".'.format(v, second_project.name))


def check_tag_mapping(first_project: Project, second_project: Project, tags_mapping: dict) -> None:
    for k, v in tags_mapping.items():
        if not first_project.meta.tag_metas.has_key(k):
            raise RuntimeError('Tag {} does not exist in input project "{}".'.format(k, first_project.name))
        if not second_project.meta.tag_metas.has_key(v):
            raise RuntimeError('Tag {} does not exist in input project "{}".'.format(v, second_project.name))


def render_labels_for_classes(labels, class_colors, canvas, missing_classes_color):
    for label in labels:
        color = class_colors.get(label.obj_class.name, missing_classes_color)
        label.geometry.draw(canvas, color)


def render_labels_for_class_name(labels, class_name, canvas):
    return render_labels_for_classes(labels, {class_name: True}, canvas, missing_classes_color=False)


def safe_ratio(num, denom):
    return (num / denom) if denom != 0 else 0


def sum_counters(elementwise_counters, counter_names):
    return {counter_name: sum(c.get(counter_name, 0) for c in elementwise_counters) for counter_name in counter_names}


def log_line(length=80, c=' '):
    logger.info(c * length)


def log_head(string):
    logger.info(string.center(80, '*'))
