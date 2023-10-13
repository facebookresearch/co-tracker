# coding: utf-8

from collections import namedtuple

import numpy as np

from supervisely.annotation.label import Label, PixelwiseScoresLabel
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.point_location import PointLocation
from supervisely.annotation.tag import Tag


def segmentation_array_to_sly_bitmaps(idx_to_class: dict, pred: np.ndarray, origin: PointLocation=None) -> list:
    """
    Converts array with segmentation results to Labels with Bitmap geometry according to idx_to_class mapping.

    Args:
        idx_to_class: Dict matching values in prediction array with appropriate ObjClass.
        pred: Array containing raw segmentation results.
        origin: Origin point for all output Bitmaps.
    return:
        A list containing result labels.
    """
    labels = []
    for cls_idx, cls_obj in idx_to_class.items():
        predicted_class_pixels = (pred == cls_idx)
        if np.any(predicted_class_pixels):
            class_geometry = Bitmap(data=predicted_class_pixels, origin=origin)
            labels.append(Label(geometry=class_geometry, obj_class=cls_obj))
    return labels


def segmentation_scores_to_per_class_labels(idx_to_class: dict, segmentation_scores: np.ndarray) -> list:
    """
    Converts output network segmentation scores into list of PixelwiseScoresLabels with MultichannelBitmap geometry.

    Args:
        idx_to_class: Dict matching channels in prediction scores array with appropriate ObjClass.
        segmentation_scores: Array containing raw segmentation scores.
    Returns:
        A list containing PixelwiseScoresLabel(s) with MultichannelBitmap(s).
    """
    return [
        PixelwiseScoresLabel(
            geometry=MultichannelBitmap(data=segmentation_scores[:, :, cls_idx, np.newaxis]),
            obj_class=cls_obj)
        for cls_idx, cls_obj in idx_to_class.items()]


DetectionNetworkPrediction = namedtuple('DetectionNetworkPrediction', ['boxes', 'scores', 'classes'])


def detection_preds_to_sly_rects(idx_to_class, network_prediction: DetectionNetworkPrediction, img_shape,
                                 min_score_threshold, score_tag_meta) -> list:
    """
    Converts network detection results to Supervisely Labels with Rectangle geometry.

    Args:
        idx_to_class: Dict matching predicted boxes with appropriate ObjClass.
        network_prediction: Network predictions packed into DetectionNetworkPrediction instance.
        img_shape: Size(height, width) of image that was used for inference.
        min_score_threshold: All detections with less scores will be dropped.
        score_tag_meta: TagMeta instance for score tags.
    Returns:
        A list containing labels with detection rectangles.
    """
    labels = []
    thr_mask = np.squeeze(network_prediction.scores) > min_score_threshold
    for box, class_id, score in zip(np.squeeze(network_prediction.boxes)[thr_mask],
                                    np.squeeze(network_prediction.classes)[thr_mask],
                                    np.squeeze(network_prediction.scores)[thr_mask]):

            xmin = round(float(box[1] * img_shape[1]))
            ymin = round(float(box[0] * img_shape[0]))
            xmax = round(float(box[3] * img_shape[1]))
            ymax = round(float(box[2] * img_shape[0]))

            rect = Rectangle(top=ymin, left=xmin, bottom=ymax, right=xmax)
            class_obj = idx_to_class[int(class_id)]
            label = Label(geometry=rect, obj_class=class_obj)

            score_tag = Tag(score_tag_meta, value=round(float(score), 4))
            label = label.add_tag(score_tag)
            labels.append(label)
    return labels
