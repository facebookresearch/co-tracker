# coding: utf-8
from collections import namedtuple

from supervisely.annotation.label import Label
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.geometry import Geometry
from supervisely.metric.common import safe_ratio

import numpy as np


LabelsPairWithScore = namedtuple('LabelsPairWithScore', ['label_1', 'label_2', 'score'])
LabelsMatchResult = namedtuple('LabelsMatchResult', ['matches', 'unmatched_labels_1', 'unmatched_labels_2'])
IndexPairWithScore = namedtuple('IndexPairWithScore', ['idx_1', 'idx_2', 'score'])
IndexMatchResult = namedtuple('IndexMatchResult', ['matches', 'unmatched_indices_1', 'unmatched_indices_2'])


def filter_labels_by_name(labels, names_whitelist):
    return [label for label in labels if label.obj_class.name in names_whitelist]


def get_iou_rect(rect: Rectangle, other: Geometry):
    maybe_other_cropped = other.crop(rect)
    if len(maybe_other_cropped) == 0:
        return 0.0
    else:
        [other_cropped] = maybe_other_cropped
        intersection_area = other_cropped.area
        if intersection_area == 0:
            return 0.0
        union_area = rect.area + other.area - intersection_area
        return intersection_area / union_area


def get_geometries_iou(geometry_1: Geometry, geometry_2: Geometry):
    if isinstance(geometry_1, Rectangle):
        return get_iou_rect(geometry_1, geometry_2)
    elif isinstance(geometry_2, Rectangle):
        return get_iou_rect(geometry_2, geometry_1)
    else:
        common_bbox = Rectangle.from_geometries_list((geometry_1, geometry_2))
        g1 = geometry_1.relative_crop(common_bbox)[0]
        g2 = geometry_2.relative_crop(common_bbox)[0]
        mask_1 = np.full(common_bbox.to_size(), False)
        g1.draw(mask_1, color=True)
        mask_2 = np.full(common_bbox.to_size(), False)
        g2.draw(mask_2, color=True)
        return safe_ratio((mask_1 & mask_2).sum(), (mask_1 | mask_2).sum())


def get_labels_iou(label_1: Label, label_2: Label, img_size=None):
    return get_geometries_iou(label_1.geometry, label_2.geometry)


def match_indices_by_score(elems_1, elems_2, score_threshold, score_fn):
    # Score all the possible pairs.
    scored_idx_pairs = [
        IndexPairWithScore(idx_1=idx_1, idx_2=idx_2, score=score_fn(elem_1, elem_2))
        for idx_1, elem_1 in enumerate(elems_1) for idx_2, elem_2 in enumerate(elems_2)]
    # Apply the threshold to avoid sorting candidates with too low scores.
    thresholded_idx_pairs = [p for p in scored_idx_pairs if p.score >= score_threshold]
    # Sort by score in descending order.
    sorted_idx_pairs = sorted(thresholded_idx_pairs, key=lambda p: p.score, reverse=True)

    # Match greedily, make sure no element is matched to more than one counterpart.
    unmatched_1 = set(range(len(elems_1)))
    unmatched_2 = set(range(len(elems_2)))
    matches = []
    for idx_pair in sorted_idx_pairs:
        if idx_pair.idx_1 in unmatched_1 and idx_pair.idx_2 in unmatched_2:
            matches.append(idx_pair)
            unmatched_1.remove(idx_pair.idx_1)
            unmatched_2.remove(idx_pair.idx_2)
    return IndexMatchResult(matches=matches, unmatched_indices_1=unmatched_1, unmatched_indices_2=unmatched_2)


def match_labels_by_iou(labels_1, labels_2, img_size, iou_threshold):
    index_matches = match_indices_by_score(labels_1, labels_2, iou_threshold, score_fn=get_labels_iou)
    return LabelsMatchResult(
        matches=[LabelsPairWithScore(label_1=labels_1[match.idx_1], label_2=labels_2[match.idx_2], score=match.score)
                 for match in index_matches.matches],
        unmatched_labels_1=[labels_1[idx] for idx in index_matches.unmatched_indices_1],
        unmatched_labels_2=[labels_2[idx] for idx in index_matches.unmatched_indices_2])
