# coding: utf-8
"""Augmentations for images and annotations"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import imgaug.augmenters.Sequential
    except:
        pass
from typing import Tuple, List, Dict, Optional

import random
import numpy as np

from supervisely.imaging import image as sly_image
from supervisely.annotation.annotation import Annotation
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.rectangle import Rectangle
from supervisely._utils import take_with_default
from supervisely.sly_logger import logger


def _validate_image_annotation_shape(img: np.ndarray, ann: Annotation) -> None:
    if img.shape[:2] != ann.img_size:
        raise RuntimeError(
            "Image shape {} doesn't match img_size {} in annotation.".format(
                img.shape[:2], ann.img_size
            )
        )


# Flips
def fliplr(img: np.ndarray, ann: Annotation) -> Tuple[np.ndarray, Annotation]:
    """
    Flips an Image and Annotation around vertical axis.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :raises: :class:`RuntimeError` if Image shape does not match img_size in Annotation
    :return: Tuple containing flipped Image and Annotation
    :rtype: :class:`Tuple[np.ndarray, Annotation]`
    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import fliplr

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        ann_info = api.annotation.download(image_id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        # Flip image and annotation
        flip_image_np, flip_ann = fliplr(image_np, ann)
    """
    _validate_image_annotation_shape(img, ann)
    res_img = sly_image.fliplr(img)
    res_ann = ann.fliplr()
    return res_img, res_ann


def flipud(img: np.ndarray, ann: Annotation) -> Tuple[np.ndarray, Annotation]:
    """
    Flips an Image and Annotation around horizontal axis.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :raises: :class:`RuntimeError` if Image shape does not match img_size in Annotation
    :return: Tuple containing flipped Image and Annotation
    :rtype: :class:`Tuple[np.ndarray, Annotation]`
    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import flipud

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        ann_info = api.annotation.download(image_id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        # Flip image and annotation
        flip_image_np, flip_ann = flipud(image_np, ann)
    """
    _validate_image_annotation_shape(img, ann)
    res_img = sly_image.flipud(img)
    res_ann = ann.flipud()
    return res_img, res_ann


# Crops
def crop(
    img: np.ndarray,
    ann: Annotation,
    top_pad: Optional[int] = 0,
    left_pad: Optional[int] = 0,
    bottom_pad: Optional[int] = 0,
    right_pad: Optional[int] = 0,
) -> Tuple[np.ndarray, Annotation]:
    """
    Crops an Image and Annotation from all sides with a given values.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :param top_pad: Top padding in pixels.
    :type top_pad: int, optional
    :param left_pad: Left padding in pixels.
    :type left_pad: int, optional
    :param bottom_pad: Bottom padding in pixels.
    :type bottom_pad: int, optional
    :param right_pad: Right padding in pixels.
    :type right_pad: int, optional
    :raises: :class:`RuntimeError` if Image shape does not match img_size in Annotation
    :return: Tuple containing cropped Image and Annotation
    :rtype: :class:`Tuple[np.ndarray, Annotation]`
    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import crop

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        print(image_np.shape)
        # Output: (800, 1067, 3)

        ann_info = api.annotation.download(image_id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        crop_image_np, crop_ann = crop(image_np, ann, top_pad=50, left_pad=100, bottom_pad=50, right_pad=100)
        print(crop_image_np.shape)
        # Output: (700, 867, 3)
    """
    _validate_image_annotation_shape(img, ann)
    height, width = img.shape[:2]
    crop_rect = Rectangle(
        top_pad, left_pad, height - bottom_pad - 1, width - right_pad - 1
    )

    res_img = sly_image.crop(img, crop_rect)
    res_ann = ann.relative_crop(crop_rect)
    return res_img, res_ann


def crop_fraction(
    img: np.ndarray,
    ann: Annotation,
    top: Optional[float] = 0,
    left: Optional[float] = 0,
    bottom: Optional[float] = 0,
    right: Optional[float] = 0,
) -> Tuple[np.ndarray, Annotation]:
    """
    Crops an Image and Annotation from all sides with the given fraction values.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :param top: Top padding in pixels.
    :type top: int, optional
    :param left: Left padding in pixels.
    :type left: int, optional
    :param bottom: Bottom padding in pixels.
    :type bottom: int, optional
    :param right: Right padding in pixels.
    :type right: int, optional
    :raises: :class:`ValueError` if fraction values not between 0 and 1
    :return: Tuple containing cropped Image and Annotation
    :rtype: :class:`Tuple[np.ndarray, Annotation]`
    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import crop_fraction

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        print(image_np.shape)
        # Output: (800, 1067, 3)

        ann_info = api.annotation.download(image_id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        crop_image_np, crop_ann = crop_fraction(image_np, ann, top=0.1, left=0.2, bottom=0.1, right=0.2)
        print(crop_image_np.shape)
        # Output: (640, 641, 3)
    """
    _validate_image_annotation_shape(img, ann)
    if not all(0 <= pad < 1 for pad in (top, left, right, bottom)):
        raise ValueError("All padding values must be between 0 and 1.")
    height, width = img.shape[:2]
    top_pixels = round(height * top)
    left_pixels = round(width * left)
    bottom_pixels = round(height * bottom)
    right_pixels = round(width * right)
    return crop(
        img,
        ann,
        top_pad=top_pixels,
        left_pad=left_pixels,
        bottom_pad=bottom_pixels,
        right_pad=right_pixels,
    )


def random_crop(
    img: np.ndarray, ann: Annotation, height: int, width: int
) -> Tuple[np.ndarray, Annotation]:
    """
    Crops an Image and Annotation at a random location.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :param height: Desired height of output crop.
    :type height: int, optional
    :param width: Desired width of output crop.
    :type width: int, optional
    :raises: :class:`RuntimeError` if Image shape does not match img_size in Annotation
    :return: Tuple containing cropped Image and Annotation
    :rtype: :class:`Tuple[np.ndarray, Annotation]`
    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import random_crop

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        print(image_np.shape)
        # Output: (800, 1067, 3)

        ann_info = api.annotation.download(image_id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        crop_image_np, crop_ann = random_crop(image_np, ann, height=500, width=700)
        print(crop_image_np.shape)
        # Output: (500, 700, 3)
    """
    _validate_image_annotation_shape(img, ann)
    img_height, img_width = img.shape[:2]

    def calc_crop_pad(old_side, crop_side):
        new_side = min(int(old_side), int(crop_side))
        min_bound = random.randint(0, old_side - new_side)  # including [a; b]
        max_bound = old_side - min_bound - new_side
        return min_bound, max_bound

    left_pad, right_pad = calc_crop_pad(img_width, width)
    top_pad, bottom_pad = calc_crop_pad(img_height, height)
    return crop(
        img,
        ann,
        top_pad=top_pad,
        left_pad=left_pad,
        bottom_pad=bottom_pad,
        right_pad=right_pad,
    )


def random_crop_fraction(
    img: np.ndarray,
    ann: Annotation,
    height_fraction_range: Tuple,
    width_fraction_range: Tuple,
) -> Tuple[np.ndarray, Annotation]:
    """
    Crops an Image and Annotation at a random location with random size in a given interval.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :param height_fraction_range: Range of relative values [0, 1] to select output height from.
    :type height_fraction_range: Tuple[float, float]
    :param width_fraction_range: Range of relative values [0, 1] to select output width from.
    :type width_fraction_range: Tuple[float, float]
    :raises: :class:`RuntimeError` if Image shape does not match img_size in Annotation
    :return: Tuple containing cropped Image and Annotation
    :rtype: :class:`Tuple[np.ndarray, Annotation]`
    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import random_crop_fraction

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        print(image_np.shape)
        # Output: (800, 1067, 3)

        ann_info = api.annotation.download(image_id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        crop_image_np, crop_ann = random_crop_fraction(image_np, ann, height_fraction_range=(0.1, 0.8), width_fraction_range=(0.1, 0.8))
        print(crop_image_np.shape)
        # Output: (486, 585, 3)
    """
    _validate_image_annotation_shape(img, ann)
    img_height, img_width = img.shape[:2]

    height_p = random.uniform(height_fraction_range[0], height_fraction_range[1])
    width_p = random.uniform(width_fraction_range[0], width_fraction_range[1])
    crop_height = round(img_height * height_p)
    crop_width = round(img_width * width_p)
    return random_crop(img, ann, height=crop_height, width=crop_width)


def batch_random_crops_fraction(
    img_ann_pairs: List[Tuple[np.ndarray, Annotation]],
    crops_per_image: int,
    height_fraction_range: Tuple,
    width_fraction_range: Tuple,
) -> List[Tuple[np.ndarray, Annotation]]:
    return [
        random_crop_fraction(img, ann, height_fraction_range, width_fraction_range)
        for img, ann in img_ann_pairs
        for _ in range(crops_per_image)
    ]


def flip_add_random_crops(
    img: np.ndarray,
    ann: Annotation,
    crops_per_image: int,
    height_fraction_range: Tuple,
    width_fraction_range: Tuple,
) -> List[Tuple[np.ndarray, Annotation]]:
    full_size_items = [(img, ann), fliplr(img, ann)]
    crops = batch_random_crops_fraction(
        full_size_items, crops_per_image, height_fraction_range, width_fraction_range
    )
    return full_size_items + crops


# TODO factor out / simplify.
def _rect_from_bounds(padding_config: Dict, img_h: int, img_w: int) -> Rectangle:
    def get_padding_pixels(raw_side, dim_name):
        side_padding_config = padding_config.get(dim_name)
        if side_padding_config is None:
            padding_pixels = 0
        elif side_padding_config.endswith("px"):
            padding_pixels = int(side_padding_config[: -len("px")])
        elif side_padding_config.endswith("%"):
            padding_fraction = float(side_padding_config[: -len("%")])
            padding_pixels = int(raw_side * padding_fraction / 100.0)
        else:
            raise ValueError(
                'Unknown padding size format: {}. Expected absolute values as "5px" or relative as "5%"'.format(
                    side_padding_config
                )
            )
        return padding_pixels

    def get_padded_side(raw_side, l_name, r_name):
        l_bound = -get_padding_pixels(raw_side, l_name)
        r_bound = raw_side + get_padding_pixels(raw_side, r_name)
        return l_bound, r_bound

    left, right = get_padded_side(img_w, "left", "right")
    top, bottom = get_padded_side(img_h, "top", "bottom")
    return Rectangle(top=top, left=left, bottom=bottom, right=right)


def instance_crop(
    img: np.ndarray,
    ann: Annotation,
    class_title: str,
    save_other_classes_in_crop: Optional[bool] = True,
    padding_config: Optional[Dict[str, str]] = None,
) -> List[Tuple[np.ndarray, Annotation]]:
    """
    Crops objects of specified classes from Image and Annotation with configurable padding.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :param class_title: Name of class to crop.
    :type class_title: str
    :param save_other_classes_in_crop: If True saves non-target classes in each cropped Annotation, otherwise don't.
    :type save_other_classes_in_crop: bool, optional
    :param padding_config: Dict with padding.
    :type padding_config: dict, optional
    :raises: :class:`ValueError` if padding size format is incorrect
    :return: List of cropped (image numpy array, Annotation) pairs
    :rtype: :class:`List[Tuple[np.ndarray, Annotation]]`

    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import instance_crop

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        print(image_np.shape)
        # Output: (800, 1067, 3)

        ann_info = api.annotation.download(image_id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        result = instance_crop(image_np, ann, 'kiwi', True, {'top': '20px', 'left': '50px', 'bottom': '700px', 'right': '1000px'})
        for crop_image_np, crop_ann in result:
            print(crop_image_np.shape)
            # Output: (270, 635, 3)
            #         (426, 345, 3)
    """
    padding_config = take_with_default(padding_config, {})
    _validate_image_annotation_shape(img, ann)
    results = []
    img_rect = Rectangle.from_size(img.shape[:2])

    if save_other_classes_in_crop:
        non_target_labels = [
            label for label in ann.labels if label.obj_class.name != class_title
        ]
    else:
        non_target_labels = []

    ann_with_non_target_labels = ann.clone(labels=non_target_labels)

    for label in ann.labels:
        if label.obj_class.name == class_title:
            src_fig_rect = label.geometry.to_bbox()
            new_img_rect = _rect_from_bounds(
                padding_config, img_w=src_fig_rect.width, img_h=src_fig_rect.height
            )
            rect_to_crop = new_img_rect.translate(src_fig_rect.top, src_fig_rect.left)
            crops = rect_to_crop.crop(img_rect)
            if len(crops) == 0:
                continue
            rect_to_crop = crops[0]
            image_crop = sly_image.crop(img, rect_to_crop)

            cropped_ann = ann_with_non_target_labels.relative_crop(rect_to_crop)

            label_crops = label.relative_crop(rect_to_crop)
            for label_crop in label_crops:
                results.append((image_crop, cropped_ann.add_label(label_crop)))
    return results


# Resize
def resize(
    img: np.ndarray, ann: Annotation, size: Tuple
) -> Tuple[np.ndarray, Annotation]:
    """
    Resizes an input Image and Annotation to a given size.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :param size: Desired size (height, width) in pixels or -1.
    :type size: Tuple[int, int]
    :raises: :class:`RuntimeError` if Image shape does not match img_size in Annotation
    :return: Tuple containing resized Image and Annotation
    :rtype: :class:`Tuple[np.ndarray, Annotation]`

    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import resize

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        print(image_np.shape)
        # Output: (800, 1067, 3)

        ann_info = api.annotation.download(image_id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        resize_image_np, resize_ann = resize(image_np, ann, (600, -1))
        print(resize_image_np.shape)
        # Output: (600, 800, 3)
    """
    _validate_image_annotation_shape(img, ann)
    height = take_with_default(size[0], -1)  # For backward capability
    width = take_with_default(size[1], -1)
    size = (height, width)

    new_size = sly_image.restore_proportional_size(in_size=ann.img_size, out_size=size)
    res_img = sly_image.resize(img, new_size)
    res_ann = ann.resize(new_size)
    return res_img, res_ann


# Resize
def scale(
    img: np.ndarray,
    ann: Annotation,
    frow: Optional[float] = None,
    fcol: Optional[float] = None,
    f: Optional[float] = None,
) -> Tuple[np.ndarray, Annotation]:
    """
    Scales an input Image and Annotation to a given size.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :param frow: Desired height scale height value.
    :type frow: float, optional
    :param fcol: Desired width scale height value.
    :type fcol: float, optional
    :param f: Desired height and width scale values in one(positive).
    :type f: float, optional
    :raises: :class:`RuntimeError` if Image shape does not match img_size in Annotation
    :return: Tuple containing scaled Image and Annotation
    :rtype: :class:`Tuple[np.ndarray, Annotation]`

    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import scale

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        print(image_np.shape)
        # Output: (800, 1067, 3)

        ann_info = api.annotation.download(193940171)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        scale_image_np, scale_ann = scale(image_np, ann, frow=0.7, fcol=0.8)
        print(scale_image_np.shape)
        # Output: (560, 854, 3)
    """
    _validate_image_annotation_shape(img, ann)
    new_size = sly_image.restore_proportional_size(
        in_size=ann.img_size, frow=frow, fcol=fcol, f=f
    )
    res_img = sly_image.resize(img, new_size)
    res_ann = ann.resize(new_size)
    return res_img, res_ann


# Rotate
class RotationModes:
    KEEP = "keep"
    CROP = "crop"


def rotate(
    img: np.ndarray,
    ann: Annotation,
    degrees: float,
    mode: Optional[str] = RotationModes.KEEP,
) -> Tuple[np.ndarray, Annotation]:  # @TODO: add "preserve_size" mode
    """
    Rotates an Image and Annotation by random angle.

    :param img: Image in numpy format, :class:`RGB`.
    :type img: np.ndarray
    :param ann: Annotation object.
    :type ann: Annotation
    :param degrees: Rotation angle.
    :type degrees: int
    :param mode: One of RotateMode enum values.
    :type mode: RotationModes, optional
    :raises: :class:`RuntimeError` if Image shape does not match img_size in Annotation
    :return: Tuple containing rotated Image and Annotation
    :rtype: :class:`Tuple[np.ndarray, Annotation]`

    :Usage Example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.aug.aug import rotate

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Download image and annotation from API
        project_id = 116501
        image_id = 193940171

        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        image_np = api.image.download_np(image_id) # <class 'numpy.ndarray'>
        print(image_np.shape)
        # Output: (800, 1067, 3)

        ann_info = api.annotation.download(image_id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)

        rotate_image_np, rotate_ann = rotate(image_np, ann, 30)
        print(rotate_image_np.shape)
        # Output: (1231, 1326, 3)
    """
    _validate_image_annotation_shape(img, ann)
    rotator = ImageRotator(img.shape[:2], degrees)

    if mode == RotationModes.KEEP:
        rect_to_crop = None

    elif mode == RotationModes.CROP:
        rect_to_crop = rotator.inner_crop

    else:
        raise NotImplementedError("Wrong black_regions mode.")

    res_img = rotator.rotate_img(img, use_inter_nearest=False)
    res_ann = ann.rotate(rotator)
    if rect_to_crop is not None:
        res_img = sly_image.crop(res_img, rect_to_crop)
        res_ann = res_ann.relative_crop(rect_to_crop)
    return res_img, res_ann


def load_imgaug(json_data: Dict) -> imgaug.augmenters.Sequential:
    try:
        import imgaug.augmenters as iaa
    except ModuleNotFoundError as e:
        logger.error(f'{e}. Try to install extra dependencies. Run "pip install supervisely[aug]"')
        raise e

    def _get_function(category_name, aug_name):
        try:
            submodule = getattr(iaa, category_name)
            aug_f = getattr(submodule, aug_name)
            return aug_f
        except Exception as e:
            logger.error(repr(e))
            raise e

    pipeline_json = json_data["pipeline"]
    random_order = json_data.get("random_order", False)

    pipeline = []
    for aug_info in pipeline_json:
        category_name = aug_info["category"]
        aug_name = aug_info["name"]
        params = aug_info["params"]

        aug_func = _get_function(category_name, aug_name)
        aug = aug_func(**params)

        sometimes = aug_info.get("sometimes", None)
        if sometimes is not None:
            aug = iaa.meta.Sometimes(sometimes, aug)
        pipeline.append(aug)
    augs = iaa.Sequential(pipeline, random_order=random_order)
    return augs
