# coding: utf-8

from supervisely.api.module_api import ApiField
from supervisely.api.pointcloud.pointcloud_tag_api import PointcloudObjectTagApi
from typing import List


class PointcloudEpisodeObjectTagApi(PointcloudObjectTagApi):
    _entity_id_field = ApiField.OBJECT_ID
    _method_bulk_add = "annotation-objects.tags.bulk.add"

    def update_frame_range(self, id: int, frame_range: List[int]) -> None:
        """Update tag frame range for annotation object in point cloud episode.

        :param id: unique ID of the tag specifically created for the object
        :type id: int
        :param frame_range: range of possible frames, it must always have strictly 2 values
        :type frame_range: List[int]
        """
        request_body = {
            ApiField.ID: id,
            ApiField.FRAME_RANGE: frame_range,
        }
        self._api.post("annotation-objects.tags.update", request_body)
