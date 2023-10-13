from supervisely.nn.inference.object_detection.object_detection import (
    ObjectDetection,
)


class PromptBasedObjectDetection(ObjectDetection):
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "prompt-based object detection"
        return info
