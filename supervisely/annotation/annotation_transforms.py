# coding: utf-8

import collections
import operator
import numpy as np
from typing import List, Callable

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.bitmap import Bitmap, SkeletonizeMethod
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.point_location import PointLocation


if not hasattr(np, 'int'): np.int = np.int_
if not hasattr(np, 'bool'): np.bool = np.bool_

def skeletonize_bitmap(ann: Annotation, classes: List[str], method_id: SkeletonizeMethod) -> Annotation:
    """
    Extracts skeletons from bitmap figures.

    Args:
        ann: Input annotation.
        classes: List of classes to skeletonize.
        method_id: Algorithm of processing. See supervisely.geometry.bitmap.SkeletonizeMethod enum.
    Returns:
        Annotation with skeletonized labels.
    """
    def _skel(label: Label):
        if label.obj_class.name not in classes:
            return [label]

        if not isinstance(label.geometry, Bitmap):
            raise RuntimeError('Input class must be a Bitmap.')

        return [label.clone(geometry=label.geometry.skeletonize(method_id))]

    return ann.transform_labels(_skel)


def approximate_vector(ann: Annotation, classes: List[str], epsilon: float) -> Annotation:
    """
    Approximates vector figures: lines and polygons.

    Args:
        ann: Input annotations.
        classes: List of classes to apply transformation.
        epsilon: Approximation accuracy (maximum distance between the original curve and its approximation)
    Returns:
        Annotation with approximated vector figures of selected classes.
    """
    def _approx(label: Label):
        if label.obj_class.name not in classes:
            return [label]

        if not isinstance(label.geometry, (Polygon, Polyline)):
            raise RuntimeError('Input class must be a Polygon or a Line.')

        return [label.clone(geometry=label.geometry.approx_dp(epsilon))]

    return ann.transform_labels(_approx)


def add_background(ann: Annotation, bg_class: ObjClass) -> Annotation:
    """
    Adds background rectangle (size equals to image size) to annotation.

    Args:
        ann: Input annotation.
        bg_class: ObjClass instance for background class label.
    Returns:
        Annotation with added background rectangle.
    """
    img_size = ann.img_size
    rect = Rectangle(0, 0, img_size[0] - 1, img_size[1] - 1)
    new_label = Label(rect, bg_class)
    return ann.add_label(new_label)


def drop_object_by_class(ann: Annotation, classes: List[str]) -> Annotation:
    """
    Removes labels of specified classes from annotation.

    Args:
        ann: Input annotation.
        classes: List of classes to remove.
    Returns:
        Annotation with removed labels of specified classes.
    """
    def _filter(label: Label):
        if label.obj_class.name in classes:
            return [label]
        return []
    return ann.transform_labels(_filter)


def filter_objects_by_area(ann: Annotation, classes: List[str], comparator=operator.lt,
                           thresh_percent: float = None) -> Annotation:  # @ TODO: add size mode
    """
    Deletes labels less (or greater) than specified percentage of image area.

    Args
        ann: Input annotation.
        classes: List of classes to filter.
        comparator: Comparison function.
        thresh_percent: Threshold percent value of image area.
    Returns:
        Annotation containing filtered labels.
    """
    imsize = ann.img_size
    img_area = float(imsize[0] * imsize[1])

    def _del_filter_percent(label: Label):
        if label.obj_class.name in classes:
            fig_area = label.area
            area_percent = 100.0 * fig_area / img_area
            if comparator(area_percent, thresh_percent):  # satisfied condition
                return []  # action 'delete'
        return [label]

    return ann.transform_labels(imsize, _del_filter_percent)


def bitwise_mask(ann: Annotation, class_mask: str, classes_to_correct: List[str],
                 bitwise_op: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.logical_and) -> Annotation:
    """
    Performs bitwise operation between two masks. Uses one target mask to correct all others.

    Args
        ann: Input annotation.
        class_mask: Class name of target mask.
        classes_to_correct: List of classes which will be corrected using target mask.
        bitwise_op: Bitwise numpy function to process masks.For example: "np.logical_or", "np.logical_and",
         "np.logical_xor".
    Returns:
        Annotation containing corrected Bitmaps.
    """
    imsize = ann.img_size

    def find_mask_class(labels, class_mask_name):
        for label in labels:
            if label.obj_class.name == class_mask_name:
                if not isinstance(label.geometry, Bitmap):
                    raise RuntimeError('Class <{}> must be a Bitmap.'.format(class_mask_name))
                return label

    mask_label = find_mask_class(ann.labels, class_mask)
    if mask_label is not None:
        target_original, target_mask = mask_label.geometry.origin, mask_label.geometry.data
        full_target_mask = np.full(imsize, False, bool)

        full_target_mask[target_original.row:target_original.row + target_mask.shape[0],
                         target_original.col:target_original.col + target_mask.shape[1]] = target_mask

        def perform_op(label):
            if label.obj_class.name not in classes_to_correct or label.obj_class.name == class_mask:
                return [label]

            if not isinstance(label.geometry, Bitmap):
                raise RuntimeError('Input class must be a Bitmap.')

            new_geom = label.geometry.bitwise_mask(full_target_mask, bitwise_op)
            return [label.clone(geometry=new_geom)] if new_geom is not None else []

        res_ann = ann.transform_labels(perform_op)
    else:
        res_ann = ann.clone()

    return res_ann


def find_contours(ann: Annotation, classes_mapping: dict) -> Annotation:  # @TODO: approximation dropped
    """

    Args:
        ann: Input annotation.
        classes_mapping: Dict matching source class names and new ObjClasses
    Returns:
        Annotation with Bitmaps converted to contours Polygons.
    """
    def to_contours(label: Label):
        new_obj_cls = classes_mapping.get(label.obj_class.name)
        if new_obj_cls is None:
            return [label]
        if not isinstance(label.geometry, Bitmap):
            raise RuntimeError('Input class must be a Bitmap.')

        return [Label(geometry=geom, obj_class=new_obj_cls) for geom in label.geometry.to_contours()]

    return ann.transform_labels(to_contours)


def extract_labels_from_mask(mask: np.ndarray, color_id_to_obj_class: dict) -> list:
    """
    Extract multiclass instances from grayscale mask and save it to labels list.
    Args:
        mask: multiclass grayscale mask
        color_id_to_obj_class: dict of objects classes assigned to color id (e.g. {1: ObjClass('cat), ...})
    Returns:
        list of labels with bitmap geometry
    """
    from skimage import measure
    from scipy import ndimage
    
    zero_offset = 1 if 0 in color_id_to_obj_class else 0
    if zero_offset > 0:
        mask = mask + zero_offset

    labeled, labels_count = measure.label(mask, connectivity=1, return_num=True)
    objects_slices = ndimage.find_objects(labeled)
    labels = []

    for object_index, slices in enumerate(objects_slices, start=1):
        crop = mask[slices]
        sub_mask = crop * (labeled[slices] == object_index).astype(np.int)

        class_index = np.max(sub_mask) - zero_offset

        if class_index in color_id_to_obj_class:
            bitmap = Bitmap(data=sub_mask.astype(np.bool), origin=PointLocation(slices[0].start, slices[1].start))
            label = Label(geometry=bitmap, obj_class=color_id_to_obj_class.get(class_index))
            labels.append(label)
    return labels
