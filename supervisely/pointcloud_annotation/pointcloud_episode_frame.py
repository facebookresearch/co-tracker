from __future__ import annotations
from typing import Optional, List, Dict

from supervisely._utils import take_with_default
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

class PointcloudEpisodeFrame(Frame):
    """
    PointcloudEpisodeFrame object for :class:`PointcloudEpisodeAnnotation<supervisely.pointcloud_annotation.pointcloud_episode_annotation.PointcloudEpisodeAnnotation>`. :class:`PointcloudEpisodeFrame<PointcloudEpisodeFrame>` object is immutable.

    :param index: Index of the PointcloudEpisodeFrame.
    :type index: int
    :param figures: List of :class:`PointcloudFigure<supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure>`.
    :type figures: list, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
        from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

        # Create pointcloud object
        obj_class_car = sly.ObjClass('car', Cuboid3d)
        pointcloud_obj_car = sly.PointcloudObject(obj_class_car)
        objects = PointcloudObjectCollection([pointcloud_obj_car])

        # Create figure
        frame_index = 7
        position, rotation, dimension = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
        cuboid = Cuboid3d(position, rotation, dimension)

        figure = sly.PointcloudFigure(pointcloud_obj_car, cuboid, frame_index=frame_index)

        frame = sly.PointcloudEpisodeFrame(frame_index, figures=[figure])
        print(frame.to_json())
        # Output: {
        #     "figures": [
        #         {
        #         "geometry": {
        #             "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
        #             "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
        #             "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
        #         },
        #         "geometryType": "cuboid_3d",
        #         "key": "cd61788d6faa401b9723f066f92a8a30",
        #         "objectKey": "c081cb9f34e54ff2bd85e04e7713ed76"
        #         }
        #     ],
        #     "index": 7
        # }
    """

    figure_type = PointcloudFigure

    def __init__(self, index: int, figures: Optional[List[PointcloudFigure]]=None):
        super().__init__(index, figures)

    @classmethod
    def from_json(
        cls, 
        data: Dict, 
        objects: PointcloudObjectCollection, 
        frames_count: Optional[int]=None, 
        key_id_map: Optional[KeyIdMap]=None
    ) -> PointcloudEpisodeFrame:
        """
        Convert a json dict to PointcloudEpisodeFrame. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Dict in json format.
        :type data: dict
        :param objects: PointcloudObjectCollection object.
        :type objects: PointcloudObjectCollection
        :param frames_count: Number of frames in point cloud.
        :type frames_count: int, optional
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :raises: :class:`ValueError` if frame index < 0 and if frame index > number of frames in point cloud
        :return: PointcloudEpisodeFrame object
        :rtype: :class:`PointcloudEpisodeFrame`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
            from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

            obj_class_car = sly.ObjClass('car', Cuboid3d)
            pointcloud_obj_car = sly.PointcloudObject(obj_class_car)
            objects = PointcloudObjectCollection([pointcloud_obj_car])

            frame_index = 7
            position, rotation, dimension = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
            cuboid = Cuboid3d(position, rotation, dimension)

            figure = sly.PointcloudFigure(pointcloud_obj_car, cuboid, frame_index=frame_index)

            frame = sly.PointcloudEpisodeFrame(frame_index, figures=[figure])
            frame_json = frame.to_json()

            frame_from_json = sly.PointcloudEpisodeFrame.from_json(frame_json, objects)
        """

        return super().from_json(data, objects, frames_count, key_id_map)

    def clone(self, index: Optional[int] = None, figures: Optional[List[PointcloudFigure]] = None) -> PointcloudEpisodeFrame:
        return super().clone(index, figures)