from typing import Dict, List, Any, Union, Optional
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.rectangle import Rectangle
from supervisely.nn.prediction_dto import PredictionKeypoints
from supervisely.annotation.label import Label
from supervisely.nn.inference.inference import Inference
from supervisely.geometry.graph import Node, KeypointsTemplate
from supervisely.project.project_meta import ProjectMeta
from supervisely.annotation.obj_class import ObjClass
from supervisely.imaging import image as sly_image
from supervisely.decorators.inference import _scale_ann_to_original_size, _process_image_path
from supervisely.io.fs import silent_remove
from supervisely.sly_logger import logger
import functools
import numpy as np

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class PoseEstimation(Inference):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[
            Union[Dict[str, Any], str]
        ] = None,  # dict with settings or path to .yml file
        keypoints_template: Optional[KeypointsTemplate] = None,
        use_gui: Optional[bool] = False,
    ):
        super().__init__(
            model_dir=model_dir,
            custom_inference_settings=custom_inference_settings,
            sliding_window_mode="none",
            use_gui=use_gui,
        )
        self.keypoints_template = keypoints_template

    def get_info(self):
        info = super().get_info()
        info["task type"] = "pose estimation"
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
        return info

    def _get_obj_class_shape(self):
        return GraphNodes

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            classes = []
            for name in self.get_classes():
                classes.append(
                    ObjClass(
                        name,
                        self._get_obj_class_shape(),
                        geometry_config=self.keypoints_template,
                    )
                )
            self._model_meta = ProjectMeta(classes)
            self._get_confidence_tag_meta()
        return self._model_meta

    def _create_label(self, dto: PredictionKeypoints):
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if obj_class is None:
            raise KeyError(
                f"Class {dto.class_name} not found in model classes {self.get_classes()}"
            )
        nodes = []
        for label, coordinate in zip(dto.labels, dto.coordinates):
            x, y = coordinate
            nodes.append(Node(label=label, row=y, col=x))
        label = Label(GraphNodes(nodes), obj_class)
        return label

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionKeypoints]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionKeypoints]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

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

        @functools.wraps(func)
        def wrapper_inference(*args, **kwargs):
            settings = kwargs["settings"]

            if "detected_bboxes" in settings:
                ann = func(*args, **kwargs)
                return ann

            rectangle_json = settings.get("rectangle")
            if rectangle_json is None:
                raise ValueError(
                    """Pose estimation task requires target object
                to be outlined in a rectangle before applying neural network."""
                )

            rectangle = Rectangle.from_json(rectangle_json)
            if "image_np" in kwargs.keys():
                image_np = kwargs["image_np"]
                if not isinstance(image_np, np.ndarray):
                    raise ValueError("Invalid input. Image path must be numpy.ndarray")
                original_image_size = image_np.shape[:2]
                image_crop_np = sly_image.crop(image_np, rectangle)
                kwargs["image_np"] = image_crop_np
                ann = func(*args, **kwargs)
                ann = _scale_ann_to_original_size(ann, original_image_size, rectangle)
            elif "image_path" in kwargs.keys():
                image_path = kwargs["image_path"]
                if not isinstance(image_path, str):
                    raise ValueError("Invalid input. Image path must be str")
                image_crop_path, original_image_size = _process_image_path(image_path, rectangle)
                kwargs["image_path"] = image_crop_path
                ann = func(*args, **kwargs)
                ann = _scale_ann_to_original_size(ann, original_image_size, rectangle)
                silent_remove(image_crop_path)
            else:
                raise ValueError("image_np or image_path not provided!")

            return ann

        return wrapper_inference

    @process_image_crop
    def _inference_image_path(
        self,
        image_path: str,
        settings: Dict,
        data_to_return: Dict,  # for decorators
    ):
        logger.debug("Input path", extra={"path": image_path})
        predictions = self.predict(image_path=image_path, settings=settings)
        ann = self._predictions_to_annotation(image_path, predictions)
        return ann
