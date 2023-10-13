import os
import threading
import time
from cacheout import Cache
from fastapi import Response, Request, status

from supervisely.geometry.bitmap import Bitmap
from supervisely.nn.prediction_dto import PredictionSegmentation
from supervisely.sly_logger import logger
from supervisely.imaging import image as sly_image
from supervisely.io.fs import silent_remove
from supervisely._utils import rand_str
from supervisely.app.content import get_data_dir
from supervisely.nn.inference import Inference
from supervisely import ProjectMeta, ObjClass, Label, env as sly_env
from supervisely.nn.inference.cache import InferenceImageCache
from supervisely.nn.inference.interactive_segmentation import functional

from typing import Dict, List, Any, Optional, Union

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class InteractiveSegmentation(Inference, InferenceImageCache):
    class Click:
        def __init__(self, x, y, is_positive):
            self.x = x
            self.y = y
            self.is_positive = is_positive

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}: {str(self.__dict__)}"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
        sliding_window_mode: Optional[Literal["basic", "advanced", "none"]] = "basic",
        use_gui: Optional[bool] = False,
    ):
        _smart_cache_ttl = sly_env.smart_cache_ttl()
        _fast_cache_ttl = max(1, _smart_cache_ttl // 2)
        Inference.__init__(self, model_dir, custom_inference_settings, sliding_window_mode, use_gui)
        InferenceImageCache.__init__(
            self,
            maxsize=sly_env.smart_cache_size(),
            ttl=_smart_cache_ttl,
            base_folder=sly_env.smart_cache_container_dir(),
        )
        self._class_names = ["mask_prediction"]
        color = [255, 0, 0]
        self._model_meta = ProjectMeta([ObjClass(self._class_names[0], Bitmap, color)])
        self._inference_image_lock = threading.Lock()
        self._inference_image_cache = Cache(ttl=_fast_cache_ttl)

        logger.debug(
            "Smart cache params",
            extra={"ttl": _smart_cache_ttl, "maxsize": sly_env.smart_cache_size()},
        )

    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "interactive segmentation"
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        info["tracking_on_videos_support"] = False
        return info

    def _get_obj_class_shape(self):
        return Bitmap

    def _create_label(self, dto: PredictionSegmentation):
        classes = self.get_classes()
        assert len(classes) == 1, "InteractiveSegmentation can't be used for multi-class inference"
        obj_class = self.model_meta.get_obj_class(classes[0])
        if not dto.mask.any():  # skip empty masks
            logger.debug(f"Mask of class {dto.class_name} is empty and will be skipped")
            return None
        geometry = Bitmap(dto.mask)
        label = Label(geometry, obj_class)
        return label

    def predict(
        self,
        image_path: str,
        clicks: List[Click],
        settings: Dict[str, Any],
    ) -> PredictionSegmentation:
        raise NotImplementedError("Have to be implemented in child class")

    def get_classes(self) -> List[str]:
        return self._class_names

    def serve(self):
        super().serve()
        server = self._app.get_server()
        self.add_cache_endpoint(server)

        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
            # 1. parse request
            # 2. download image
            # 3. make crop
            # 4. predict

            logger.debug(
                f"smart_segmentation inference: context=",
                extra={**request.state.context, "api_token": "***"},
            )

            try:
                state = request.state.state
                settings = self._get_inference_settings(state)
                smtool_state = request.state.context
                api = request.state.api
                crop = smtool_state["crop"]
                positive_clicks, negative_clicks = (
                    smtool_state["positive"],
                    smtool_state["negative"],
                )
                if len(positive_clicks) + len(negative_clicks) == 0:
                    logger.warn("No clicks received.")
                    response = {
                        "origin": None,
                        "bitmap": None,
                        "success": True,
                        "error": None,
                    }
                    return response
            except Exception as exc:
                logger.warn("Error parsing request:" + str(exc), exc_info=True)
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "400: Bad request.", "success": False}

            # collect clicks
            clicks = [{**click, "is_positive": True} for click in positive_clicks]
            clicks += [{**click, "is_positive": False} for click in negative_clicks]
            clicks = functional.transform_clicks_to_crop(crop, clicks)
            is_in_bbox = functional.validate_click_bounds(crop, clicks)
            if not is_in_bbox:
                logger.warn(f"Invalid value: click is out of bbox bounds.")
                return {
                    "origin": None,
                    "bitmap": None,
                    "success": True,
                    "error": None,
                }

            # download image if needed (using cache)
            app_dir = get_data_dir()
            hash_str = functional.get_hash_from_context(smtool_state)
            if hash_str not in self._inference_image_cache:
                logger.debug(f"downloading image: {hash_str}")
                image_np = functional.download_image_from_context(
                    smtool_state,
                    api,
                    app_dir,
                    self.download_image,
                    self.download_frame,
                    self.download_image_by_hash,
                )
                self._inference_image_cache.set(hash_str, image_np)
            else:
                logger.debug(f"image found in cache: {hash_str}")
                image_np = self._inference_image_cache.get(hash_str)

            # crop
            image_np = functional.crop_image(crop, image_np)
            image_path = os.path.join(app_dir, f"{time.time()}_{rand_str(10)}.jpg")
            sly_image.write(image_path, image_np)

            self._inference_image_lock.acquire()
            try:
                # predict
                logger.debug(f"predict: {smtool_state['request_uid']}")
                clicks_to_predict = [self.Click(c["x"], c["y"], c["is_positive"]) for c in clicks]
                pred_mask = self.predict(image_path, clicks_to_predict, settings).mask
            finally:
                logger.debug(f"predict done: {smtool_state['request_uid']}")
                self._inference_image_lock.release()
                silent_remove(image_path)

            if pred_mask.any():
                bitmap = Bitmap(pred_mask)
                bitmap_origin, bitmap_data = functional.format_bitmap(bitmap, crop)
                logger.debug(f"smart_segmentation inference done!")
                response = {
                    "origin": bitmap_origin,
                    "bitmap": bitmap_data,
                    "success": True,
                    "error": None,
                }
            else:
                logger.debug(f"Predicted mask is empty.")
                response = {
                    "origin": None,
                    "bitmap": None,
                    "success": True,
                    "error": None,
                }
            return response
