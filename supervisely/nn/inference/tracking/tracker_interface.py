import numpy as np
from typing import Generator, Optional, List, Callable, OrderedDict, Dict
from collections import OrderedDict

import supervisely as sly
from supervisely.geometry.geometry import Geometry
from logging import Logger


class TrackerInterface:
    def __init__(
        self,
        context,
        api,
        load_all_frames=False,
        notify_in_predict=False,
        per_point_polygon_tracking=True,
        frame_loader: Callable[[sly.Api, int, int], np.ndarray] = None,
    ):
        self.api: sly.Api = api
        self.logger: Logger = api.logger
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.object_ids = list(context["objectIds"])
        self.figure_ids = list(context["figureIds"])
        self.direction = context["direction"]

        # all geometries
        self.stop = len(self.figure_ids) * self.frames_count
        self.global_pos = 0
        self.global_stop_indicatior = False

        self.geometries: OrderedDict[int, Geometry] = OrderedDict()
        self.frames_indexes: List[int] = []
        self._cur_frames_indexes: List[int] = []
        self._frames: Optional[np.ndarray] = None
        self.load_all_frames = load_all_frames
        self.per_point_polygon_tracking = per_point_polygon_tracking

        # increase self.stop by num of frames will be loaded
        self._add_frames_indexes()

        # increase self.stop by num of points
        self._add_geometries()

        self._hot_cache: Dict[int, np.ndarray] = {}
        self._local_cache_loader = frame_loader

        if self.load_all_frames:
            if notify_in_predict:
                self.stop += self.frames_count + 1
            self._load_frames()

    def add_object_geometries(self, geometries: List[Geometry], object_id: int, start_fig: int):
        for frame, geometry in zip(self._cur_frames_indexes[1:], geometries):
            if self.global_stop_indicatior:
                self._notify(True, task="stop tracking")
                break
            self.add_object_geometry_on_frame(geometry, object_id, frame)

        self.geometries[start_fig] = geometries[-1]

    def frames_loader_generator(self) -> Generator[None, None, None]:
        if self.load_all_frames:
            self._cur_frames_indexes = self.frames_indexes
            yield
            return

        ind = self.frames_indexes[0]
        frame = self._load_frame(ind)
        for next_ind in self.frames_indexes[1:]:
            next_frame = self._load_frame(next_ind)
            self._frames = np.array([frame, next_frame])
            self.frames_count = 1
            self._cur_frames_indexes = [ind, next_ind]
            yield
            frame = next_frame
            ind = next_ind

            if self.global_stop_indicatior:
                self.clear_cache()
                return

    def add_object_geometry_on_frame(
        self,
        geometry: Geometry,
        object_id: int,
        frame_ind: int,
        notify: bool = True,
    ):
        self.api.video.figure.create(
            self.video_id,
            object_id,
            frame_ind,
            geometry.to_json(),
            geometry.geometry_name(),
            self.track_id,
        )
        self.logger.debug(f"Added {geometry.geometry_name()} to frame #{frame_ind}")
        if notify:
            self._notify(task="add geometry on frame")

    def clear_cache(self):
        self._hot_cache.clear()

    def _add_geometries(self):
        self.logger.info("Adding geometries.")
        points = 0
        for figure_id in self.figure_ids:
            figure = self.api.video.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries[figure_id] = geometry

            self.api.logger.debug(f"Added {figure.geometry_type} #{figure_id}")

            # per point track notification
            if isinstance(geometry, sly.Point):
                points += 1
            elif isinstance(geometry, sly.Polygon):
                points += len(geometry.exterior) + len(geometry.interior)
            elif isinstance(geometry, sly.GraphNodes):
                points += len(geometry.nodes.items())
            elif isinstance(geometry, sly.Polyline):
                points += len(geometry.exterior)

        if self.per_point_polygon_tracking:
            if not self.load_all_frames:
                self.stop += points * self.frames_count
            else:
                self.stop += points

        self.logger.info("Geometries added.")
        # TODO: other geometries

    def _add_frames_indexes(self):
        total_frames = self.api.video.get_info_by_id(self.video_id).frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += 1 if self.direction == "forward" else -1

        if self.load_all_frames:
            self.stop += len(self.frames_indexes)

    def _load_frame(self, frame_index):
        if frame_index in self._hot_cache:
            return self._hot_cache[frame_index]
        if self._local_cache_loader is None:
            self._hot_cache[frame_index] = self.api.video.frame.download_np(
                self.video_id, frame_index
            )
        else:
            self._hot_cache[frame_index] = self._local_cache_loader(
                self.api, self.video_id, frame_index
            )
        return self._hot_cache[frame_index]

    def _load_frames(self):
        rgbs = []
        self.logger.info(f"Loading {len(self.frames_indexes)} frames.")

        for frame_index in self.frames_indexes:
            img_rgb = self._load_frame(frame_index)
            rgbs.append(img_rgb)
            self._notify(task="load frame")
        self._frames = rgbs
        self.logger.info("Frames loaded.")

    def _notify(
        self,
        stop: bool = False,
        fstart: Optional[int] = None,
        fend: Optional[int] = None,
        task: str = "not defined",
    ):
        self.global_pos += 1

        if stop:
            pos = self.stop
        else:
            pos = self.global_pos

        fstart = min(self.frames_indexes) if fstart is None else fstart
        fend = max(self.frames_indexes) if fend is None else fend

        self.logger.debug(f"Task: {task}")
        self.logger.debug(f"Notification status: {pos}/{self.stop}")

        self.global_stop_indicatior = self.api.video.notify_progress(
            self.track_id,
            self.video_id,
            fstart,
            fend,
            pos,
            self.stop,
        )

        self.logger.debug(f"Notification status: stop={self.global_stop_indicatior}")

        if self.global_stop_indicatior and self.global_pos < self.stop:
            self.logger.info("Task stoped by user.")

    @property
    def frames(self) -> np.ndarray:
        return self._frames

    @property
    def frames_with_notification(self) -> np.ndarray:
        """Use this in prediction."""
        self._notify(task="get frames")
        return self._frames
