import supervisely as sly
from logging import Logger


class Tracker3DInterface:
    def __init__(
        self,
        context,
        api,
    ):
        self.api: sly.Api = api
        self.logger: Logger = api.logger
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.dataset_id = context["datasetId"]
        self.pc_ids = (
            context["pointCloudIds"]
            if context["direction"] == "forward"
            else context["pointCloudIds"][::-1]
        )
        self.figure_ids = context["figureIds"]
        self.object_ids = context["objectIds"]
        self.direction = context["direction"]

        self.stop = (
            len(self.pc_ids)
            + (len(self.figure_ids) * len(self.pc_ids))
            + (len(self.figure_ids) * self.frames_count)
        )
        self.global_pos = 0
        self.global_stop_indicatior = False
        self.pc_dir = "./pointclouds/"

        self.geometries = []
        self.frames_indexes = []

        self.add_frames_indexes()
        self.add_geometries()

        self.logger.info("Tracker 3D interface initialized")

    def add_geometries(self):
        for figure_id in self.figure_ids:
            figure = self.api.pointcloud.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries.append(geometry)

    def add_frames_indexes(self):
        total_frames = len(self.api.pointcloud_episode.get_frame_name_map(self.dataset_id))
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count:
            self.frames_indexes.append(cur_index)
            cur_index += 1 if self.direction == "forward" else -1

    def _notify(
        self,
        stop: bool = False,
        task: str = "not defined",
    ):
        self.global_pos += 1

        if stop:
            pos = self.stop
        else:
            pos = self.global_pos

        self.logger.debug(f"Task: {task}")
        self.logger.debug(f"Notification status: {pos}/{self.stop}")

        self.global_stop_indicatior = self.api.pointcloud_episode.notify_progress(
            self.track_id,
            self.dataset_id,
            self.pc_ids,
            pos,
            self.stop,
        )

        if self.global_stop_indicatior and self.global_pos < self.stop:
            self.logger.info("Task stoped by user")

    def add_cuboid_on_frame(self, pcd_id, object_id, cuboid_json, track_id):
        self.api.pointcloud_episode.figure.create(
            pcd_id, object_id, cuboid_json, "cuboid_3d", track_id
        )
        self._notify(task="add geometry on frame")
