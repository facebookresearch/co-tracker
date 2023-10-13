import numpy as np
import functools
from fastapi import Request, BackgroundTasks
from typing import Any, Dict, Optional, Union

import supervisely as sly
import supervisely.nn.inference.tracking.functional as F
from supervisely.annotation.label import Label
from supervisely.nn.prediction_dto import PredictionSegmentation
from supervisely.nn.inference.tracking.tracker_interface import TrackerInterface
from supervisely.nn.inference.cache import InferenceImageCache
from supervisely.nn.inference import Inference


class MaskTracking(Inference, InferenceImageCache):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
    ):
        Inference.__init__(
            self,
            model_dir,
            custom_inference_settings,
            sliding_window_mode=None,
            use_gui=False,
        )
        InferenceImageCache.__init__(
            self,
            maxsize=sly.env.smart_cache_size(),
            ttl=sly.env.smart_cache_ttl(),
            is_persistent=True,
            base_folder=sly.env.smart_cache_container_dir(),
        )

        try:
            self.load_on_device(model_dir, "cuda")
        except RuntimeError:
            self.load_on_device(model_dir, "cpu")
            sly.logger.warn("Failed to load model on CUDA device.")

        sly.logger.debug(
            "Smart cache params",
            extra={
                "ttl": sly.env.smart_cache_ttl(),
                "maxsize": sly.env.smart_cache_size(),
            },
        )

    def get_info(self):
        info = super().get_info()
        info["task type"] = "tracking"
        return info

    def serve(self):
        super().serve()
        server = self._app.get_server()
        self.add_cache_endpoint(server)

        @server.post("/track")
        def start_track(request: Request, task: BackgroundTasks):
            task.add_task(track, request)
            return {"message": "Track task started."}

        def send_error_data(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                value = None
                try:
                    value = func(*args, **kwargs)
                except Exception as exc:
                    # print("An error occured:")
                    # print(traceback.format_exc())
                    request: Request = args[0]
                    context = request.state.context
                    api: sly.Api = request.state.api
                    track_id = context["trackId"]
                    api.logger.error("An error occured:")
                    api.logger.exception(exc)

                    api.post(
                        "videos.notify-annotation-tool",
                        data={
                            "type": "videos:tracking-error",
                            "data": {
                                "trackId": track_id,
                                "error": {"message": repr(exc)},
                            },
                        },
                    )
                return value

            return wrapper

        @send_error_data
        def track(request: Request):
            context = request.state.context
            api: sly.Api = request.state.api

            self.video_interface = TrackerInterface(
                context=context,
                api=api,
                load_all_frames=True,
                notify_in_predict=True,
                per_point_polygon_tracking=False,
                frame_loader=self.download_frame,
            )
            api.logger.info("Starting tracking process")
            # load frames
            frames = self.video_interface.frames
            # combine several binary masks into one multilabel mask
            i = 0
            label2id = {}

            for (fig_id, geometry), obj_id in zip(
                self.video_interface.geometries.items(),
                self.video_interface.object_ids,
            ):
                original_geometry = geometry.clone()
                if not isinstance(geometry, sly.Bitmap) and not isinstance(geometry, sly.Polygon):
                    raise TypeError(
                        f"This app does not support {geometry.geometry_name()} tracking"
                    )
                # convert polygon to bitmap
                if isinstance(geometry, sly.Polygon):
                    polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                    polygon_label = sly.Label(geometry, polygon_obj_class)
                    bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                    bitmap_label = polygon_label.convert(bitmap_obj_class)[0]
                    geometry = bitmap_label.geometry
                if i == 0:
                    multilabel_mask = geometry.data.astype(int)
                    multilabel_mask = np.zeros(frames[0].shape, dtype=np.uint8)
                    geometry.draw(bitmap=multilabel_mask, color=[1, 1, 1])
                    i += 1
                else:
                    i += 1
                    geometry.draw(bitmap=multilabel_mask, color=[i, i, i])
                label2id[i] = {
                    "fig_id": fig_id,
                    "obj_id": obj_id,
                    "original_geometry": original_geometry.geometry_name(),
                }
            # run tracker
            tracked_multilabel_masks = self.predict(
                frames=frames, input_mask=multilabel_mask[:, :, 0]
            )
            tracked_multilabel_masks = np.array(tracked_multilabel_masks)
            # decompose multilabel masks into binary masks
            for i in np.unique(tracked_multilabel_masks):
                if i != 0:
                    binary_masks = tracked_multilabel_masks == i
                    fig_id = label2id[i]["fig_id"]
                    obj_id = label2id[i]["obj_id"]
                    geometry_type = label2id[i]["original_geometry"]
                    for j, mask in enumerate(binary_masks[1:]):
                        # check if mask is not empty
                        if not np.any(mask):
                            api.logger.info(
                                f"Skipping empty mask on frame {self.video_interface.frame_index + j + 1}"
                            )
                            # update progress bar anyway (otherwise it will not be finished)
                            self.video_interface._notify(task="add geometry on frame")
                        else:
                            if geometry_type == "polygon":
                                bitmap_geometry = sly.Bitmap(mask)
                                bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                                bitmap_label = sly.Label(bitmap_geometry, bitmap_obj_class)
                                polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                                polygon_labels = bitmap_label.convert(polygon_obj_class)
                                geometries = [label.geometry for label in polygon_labels]
                            else:
                                geometries = [sly.Bitmap(mask)]
                            for l, geometry in enumerate(geometries):
                                if l == len(geometries) - 1:
                                    notify = True
                                else:
                                    notify = False
                                self.video_interface.add_object_geometry_on_frame(
                                    geometry,
                                    obj_id,
                                    self.video_interface.frames_indexes[j + 1],
                                    notify=notify,
                                )
                    if self.video_interface.global_stop_indicatior:
                        return
                    api.logger.info(f"Figure with id {fig_id} was successfully tracked")
