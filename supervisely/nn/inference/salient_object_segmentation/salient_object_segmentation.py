from typing import Dict
from supervisely.geometry.bitmap import Bitmap
from supervisely.nn.prediction_dto import PredictionMask
from supervisely.annotation.label import Label
from supervisely.sly_logger import logger
import numpy as np
import functools
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.decorators.inference import _scale_ann_to_original_size, _process_image_path
from supervisely.io.fs import silent_remove
from supervisely.decorators.inference import process_image_sliding_window
from supervisely.nn.inference.semantic_segmentation.semantic_segmentation import (
    SemanticSegmentation,
)


class SalientObjectSegmentation(SemanticSegmentation):
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "salient object segmentation"
        return info

    def _create_label(self, dto: PredictionMask):
        geometry = Bitmap(dto.mask)
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if not dto.mask.any():  # skip empty masks
            logger.debug(f"Mask is empty and will be slipped")
            return None
        label = Label(geometry, obj_class)
        return [label]

    def process_image_crop(func):
        """
        Decorator for processing annotation labels before and after inference.
        Crops input image before inference if kwargs['state']['rectangle_crop'] provided
        and then scales annotation back to original image size.
        Keyword arguments:
        :param image_np: Image in numpy.ndarray format (use image_path or image_np, not both)
        :type image_np: numpy.ndarray
        :param image_path: Path to image (use image_path or image_np, not both)
        :type image_path: str
        :raises: :class:`ValueError`, if image_np or image_path invalid or not provided
        :return: Annotation in json format
        :rtype: :class:`dict`
        """

        # function for bounding boxes padding
        def bbox_padding(rectangle, padding):
            padding = padding.strip()  # remove blank spaces
            if padding.endswith("px"):
                padding = int(padding[:-2])
                format = "pixels"
            elif padding.endswith("%"):
                padding = int(padding[:-1])
                padding = round(padding / 100, 2)  # from % to float
                format = "percentages"
            else:
                raise ValueError(
                    "Unsupported padding unit: only pixels (e.g. 10px) and percentages (e.g. 10%) are supported"
                )
            if padding < 0:
                padding = 0
            left, right, top, bottom = (
                rectangle.left,
                rectangle.right,
                rectangle.top,
                rectangle.bottom,
            )
            if format == "pixels":
                pad_left = left - padding
                pad_right = right + padding
                pad_top = top - padding
                pad_bottom = bottom + padding
            elif format == "percentages":
                width, height = rectangle.width, rectangle.height
                width_padding = int(width * padding)
                height_padding = int(height * padding)
                pad_left = left - width_padding
                pad_right = right + width_padding
                pad_top = top - height_padding
                pad_bottom = bottom + height_padding
            return Rectangle(pad_top, pad_left, pad_bottom, pad_right)

        # function for padded bounding boxes processing
        def process_padded_bbox(image, padded_bbox):
            img_height, img_width = image.shape[:2]
            x_min, y_min, x_max, y_max = 0, 0, img_width, img_height
            box_x_min, box_y_min, box_x_max, box_y_max = (
                padded_bbox.left,
                padded_bbox.top,
                padded_bbox.right,
                padded_bbox.bottom,
            )
            if box_x_min <= x_min:
                box_x_min = x_min + 1
            if box_y_min <= y_min:
                box_y_min = y_min + 1
            if box_x_max >= x_max:
                box_x_max = x_max - 1
            if box_y_max >= y_max:
                box_y_max = y_max - 1
            return Rectangle(box_y_min, box_x_min, box_y_max, box_x_max)

        @functools.wraps(func)
        def wrapper_inference(*args, **kwargs):
            settings = kwargs["settings"]
            rectangle_json = settings.get("rectangle")

            if rectangle_json is None:
                ann = func(*args, **kwargs)
                return ann

            rectangle = Rectangle.from_json(rectangle_json)
            padding = settings.get("bbox_padding")
            if padding is not None:
                original_rectangle = rectangle
                rectangle = bbox_padding(rectangle, padding)
                if "image_np" in kwargs.keys():
                    image_np = kwargs["image_np"]
                elif "image_path" in kwargs.keys():
                    image_path = kwargs["image_path"]
                    image_np = sly_image.read(image_path)
                rectangle = process_padded_bbox(image_np, rectangle)

            if "image_np" in kwargs.keys():
                image_np = kwargs["image_np"]
                if not isinstance(image_np, np.ndarray):
                    raise ValueError("Invalid input. Image path must be numpy.ndarray")
                original_image_size = image_np.shape[:2]
                image_crop_np = sly_image.crop(image_np, rectangle)
                kwargs["image_np"] = image_crop_np
                ann = func(*args, **kwargs)
                ann = _scale_ann_to_original_size(ann, original_image_size, rectangle)
                if padding:
                    ann = ann.crop_labels(
                        original_rectangle
                    )  # crop labels to avoid overlapping masks
            elif "image_path" in kwargs.keys():
                image_path = kwargs["image_path"]
                if not isinstance(image_path, str):
                    raise ValueError("Invalid input. Image path must be str")
                image_crop_path, original_image_size = _process_image_path(image_path, rectangle)
                kwargs["image_path"] = image_crop_path
                ann = func(*args, **kwargs)
                ann = _scale_ann_to_original_size(ann, original_image_size, rectangle)
                silent_remove(image_crop_path)
                if padding:
                    ann = ann.crop_labels(
                        original_rectangle
                    )  # crop labels to avoid overlapping masks
            else:
                raise ValueError("image_np or image_path not provided!")

            return ann

        return wrapper_inference

    @process_image_sliding_window
    @process_image_crop
    def _inference_image_path(
        self,
        image_path: str,
        settings: Dict,
        data_to_return: Dict,  # for decorators
    ):
        inference_mode = settings.get("inference_mode", "full_image")
        logger.debug(
            "Inferring image_path:", extra={"inference_mode": inference_mode, "path": image_path}
        )

        if inference_mode == "sliding_window" and settings["sliding_window_mode"] == "advanced":
            predictions = self.predict_raw(image_path=image_path, settings=settings)
        else:
            predictions = self.predict(image_path=image_path, settings=settings)
        ann = self._predictions_to_annotation(image_path, predictions)

        logger.debug(
            f"Inferring image_path done. pred_annotation:",
            extra=dict(w=ann.img_size[1], h=ann.img_size[0], n_labels=len(ann.labels)),
        )
        return ann
