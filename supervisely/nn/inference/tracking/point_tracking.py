import numpy as np
import functools
from fastapi import Request, BackgroundTasks
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import supervisely as sly
import supervisely.nn.inference.tracking.functional as F
from supervisely.annotation.label import Label
from supervisely.nn.prediction_dto import Prediction, PredictionPoint
from supervisely.nn.inference.tracking.tracker_interface import TrackerInterface
from supervisely.nn.inference import Inference
from supervisely.nn.inference.cache import InferenceImageCache


class PointTracking(Inference, InferenceImageCache):
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
                "path": sly.env.smart_cache_container_dir(),
            },
        )

    def get_info(self):
        info = super().get_info()
        info["task type"] = "tracking"
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
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

            if self.custom_inference_settings_dict.get("load_all_frames"):
                load_all_frames = True
            else:
                load_all_frames = False
            video_interface = TrackerInterface(
                context=context,
                api=api,
                load_all_frames=load_all_frames,
                frame_loader=self.download_frame,
            )

            range_of_frames = [
                video_interface.frames_indexes[0],
                video_interface.frames_indexes[-1],
            ]

            self.run_cache_task_manually(
                api,
                [range_of_frames],
                video_id=video_interface.video_id,
            )
            api.logger.info("Start tracking.")

            for _ in video_interface.frames_loader_generator():
                for (fig_id, geom), obj_id in zip(
                    video_interface.geometries.items(),
                    video_interface.object_ids,
                ):
                    if isinstance(geom, sly.Point):
                        geometries = self._predict_point_geometries(
                            geom,
                            video_interface,
                        )
                    elif isinstance(geom, sly.Polygon):
                        if len(geom.interior) > 0:
                            raise ValueError("Can't track polygons with iterior.")
                        geometries = self._predict_polygon_geometries(
                            geom,
                            video_interface,
                        )
                    elif isinstance(geom, sly.GraphNodes):
                        geometries = self._predict_graph_geometries(
                            geom,
                            video_interface,
                        )
                    elif isinstance(geom, sly.Polyline):
                        geometries = self._predict_polyline_geometries(
                            geom,
                            video_interface,
                        )
                    else:
                        raise TypeError(f"Tracking does not work with {geom.geometry_name()}.")

                    video_interface.add_object_geometries(geometries, obj_id, fig_id)
                    api.logger.info(f"Object #{obj_id} tracked.")

                    if video_interface.global_stop_indicatior:
                        return

    def predict(
        self,
        rgb_images: List[np.ndarray],
        settings: Dict[str, Any],
        start_object: PredictionPoint,
    ) -> List[PredictionPoint]:
        """
        Track point on given frames.

        :param rgb_images: RGB frames, `m` frames
        :type rgb_images: List[np.array]
        :param settings: model parameters
        :type settings: Dict[str, Any]
        :param start_objects: point to track on the initial frame
        :type start_objects: PredictionPoint
        :return: predicted points for frame range (0, m]; `m-1` prediction in total
        :rtype: List[PredictionPoint]
        """
        raise NotImplementedError

    def visualize(
        self,
        predictions: List[PredictionPoint],
        images: List[np.ndarray],
        vis_path: str,
        thickness: int = 2,
    ):
        vis_path = Path(vis_path)

        for i, (pred, image) in enumerate(zip(predictions, images)):
            out_path = vis_path / f"img_{i}.jpg"
            ann = self._predictions_to_annotation(image, [pred])
            ann.draw_pretty(
                bitmap=image,
                color=(255, 0, 0),
                thickness=thickness,
                output_path=str(out_path),
                fill_rectangles=False,
            )

    def _create_label(self, dto: PredictionPoint) -> sly.Point:
        geometry = sly.Point(row=dto.row, col=dto.col)
        return Label(geometry, sly.ObjClass("", sly.Point))

    def _get_obj_class_shape(self):
        return sly.Point

    def _predict_point_geometries(
        self,
        geom: sly.Point,
        interface: TrackerInterface,
    ) -> List[sly.Point]:
        pp_geom = PredictionPoint("point", col=geom.col, row=geom.row)
        predicted: List[Prediction] = self.predict(
            interface.frames_with_notification,
            self.custom_inference_settings_dict,
            pp_geom,
        )
        return F.dto_points_to_sly_points(predicted)

    def _predict_polygon_geometries(
        self,
        geom: sly.Polygon,
        interface: TrackerInterface,
    ) -> List[sly.Polygon]:
        polygon_points = F.numpy_to_dto_point(geom.exterior_np, "polygon")
        exterior_per_time = [[] for _ in range(interface.frames_count)]

        for pp_geom in polygon_points:
            points: List[Prediction] = self.predict(
                interface.frames_with_notification,
                self.custom_inference_settings_dict,
                pp_geom,
            )
            points_loc = F.dto_points_to_point_location(points)
            for fi, point_loc in enumerate(points_loc):
                exterior_per_time[fi].append(point_loc)

        return F.exteriors_to_sly_polygons(exterior_per_time)

    def _predict_graph_geometries(
        self,
        geom: sly.GraphNodes,
        interface: TrackerInterface,
    ) -> sly.GraphNodes:
        nodes_per_time = [[] for _ in range(interface.frames_count)]
        points_with_id = F.graph_to_dto_points(geom)

        for point, pid in zip(*points_with_id):
            preds: List[PredictionPoint] = self.predict(
                interface.frames_with_notification,
                self.custom_inference_settings_dict,
                point,
            )
            nodes = F.dto_points_to_sly_nodes(preds, pid)

            for time, node in enumerate(nodes):
                nodes_per_time[time].append(node)

        return F.nodes_to_sly_graph(nodes_per_time)

    def _predict_polyline_geometries(
        self,
        geom: sly.Polyline,
        interface: TrackerInterface,
    ) -> List[sly.Polyline]:
        polyline_points = F.numpy_to_dto_point(geom.exterior_np, "polyline")
        lines_per_time = [[] for _ in range(interface.frames_count)]

        for point in polyline_points:
            preds: List[PredictionPoint] = self.predict(
                interface.frames_with_notification,
                self.custom_inference_settings_dict,
                point,
            )
            sly_points_loc = F.dto_points_to_point_location(preds)

            for time, point_loc in enumerate(sly_points_loc):
                lines_per_time[time].append(point_loc)

        return F.exterior_to_sly_polyline(lines_per_time)

    def _predictions_to_annotation(
        self, image: np.ndarray, predictions: List[Prediction]
    ) -> sly.Annotation:
        labels = []
        for prediction in predictions:
            label = self._create_label(prediction)
            if label is None:
                # for example empty mask
                continue
            if isinstance(label, list):
                labels.extend(label)
                continue
            labels.append(label)

        # create annotation with correct image resolution
        ann = sly.Annotation(img_size=image.shape[:2])
        ann = ann.add_labels(labels)
        return ann
