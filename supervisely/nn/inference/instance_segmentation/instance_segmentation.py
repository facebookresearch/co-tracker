from typing import Dict, List, Any, Union
from supervisely.geometry.bitmap import Bitmap, Rectangle
from supervisely.nn.prediction_dto import PredictionMask, PredictionBBox
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.sly_logger import logger
from supervisely.nn.inference.inference import Inference


class InstanceSegmentation(Inference):
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "instance segmentation"
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
        return info

    def _get_obj_class_shape(self):
        return Bitmap

    def _create_label(self, dto: Union[PredictionMask, PredictionBBox]):
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if obj_class is None:
            raise KeyError(
                f"Class {dto.class_name} not found in model classes {self.get_classes()}"
            )
        if isinstance(dto, PredictionMask):
            if not dto.mask.any():  # skip empty masks
                logger.debug(f"Mask of class {dto.class_name} is empty and will be skipped")
                return None
            geometry = Bitmap(dto.mask)
        elif isinstance(dto, PredictionBBox):
            geometry = Rectangle(*dto.bbox_tlbr)
        tags = []
        if dto.score is not None:
            tags.append(Tag(self._get_confidence_tag_meta(), dto.score))
        label = Label(geometry, obj_class, tags)
        return label

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[Union[PredictionMask, PredictionBBox]]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[Union[PredictionMask, PredictionBBox]]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )
