# coding: utf-8

from supervisely.api.module_api import ApiField
from supervisely.api.entity_annotation.tag_api import TagApi
from typing import List, Optional, Union


class PointcloudTagApi(TagApi):
    """
    :class:`PointcloudTag<supervisely.pointcloud_annotation.pointcloud_tag.PointcloudTag>` for point clouds. :class:`PointcloudTagApi<PointcloudTagApi>` object is immutable.
    """

    _entity_id_field = ApiField.ENTITY_ID
    _method_bulk_add = "point-clouds.tags.bulk.add"

    def add(
        self,
        tag_meta_id: int,
        pointcloud_id: int,
        value: Optional[Union[str, int]] = None,
    ) -> int:
        """Add tag to point cloud.

        :param tag_id: TagMeta ID in project `tag_metas`
        :type tag_id: int
        :param pointcloud_id: Point cloud ID
        :type pointcloud_id: int
        :param value: possible_values from TagMeta, defaults to None
        :type value: Optional[Union[str, int]], optional
        :return: ID of the tag assigned to the point cloud
        :rtype: int
        """
        request_body = {
            ApiField.TAG_ID: tag_meta_id,
            ApiField.ENTITY_ID: pointcloud_id,
        }
        if value:
            request_body[ApiField.VALUE] = value

        response = self._api.post("point-clouds.tags.add", request_body)
        id = response.json()[ApiField.ID]
        return id

    def remove(self, tag_id: int) -> None:
        """Remove tag from point cloud.

        :param tag_id: tag ID of certain point cloud
        :type tag_id: int
        """
        request_body = {ApiField.ID: tag_id}
        self._api.post("point-clouds.tags.remove", request_body)

    def update(self, tag_id: int, value: Union[str, int]) -> None:
        """Update tag value for point cloud.
        You could use only those values, which are correspond to TagMeta `value_type` and `possible_values`

        :param tag_id: tag ID of certain object
        :type tag_id: int
        :param value: possible_values from TagMeta
        :type value: Union[str, int]
        """
        request_body = {
            ApiField.ID: tag_id,
            ApiField.VALUE: value,
        }
        self._api.post("point-clouds.tags.update-value", request_body)


class PointcloudObjectTagApi(TagApi):
    _entity_id_field = ApiField.OBJECT_ID
    _method_bulk_add = "annotation-objects.tags.bulk.add"

    def add(
        self,
        tag_meta_id: int,
        object_id: int,
        value: Optional[Union[str, int]] = None,
        frame_range: Optional[List[int]] = None,
    ) -> int:
        """Add a tag to an annotation object.
        It is possible to add a `value` as an option for point clouds and point cloud episodes.
        It is also possible to add `frame_range` as an option, but only for point cloud episodes.

        :param tag_meta_id: TagMeta ID in project `tag_metas`
        :type tag_meta_id: int
        :param object_id: Object ID in project annotation objects
        :type object_id: int
        :param value: possible_values from TagMeta, defaults to None
        :type value: Optional[Union[str, int]], optional
        :param frame_range: array of 2 frame numbers in point cloud episodes, defaults to None
        :type frame_range: Optional[List[int]], optional
        :return: ID of the tag assigned to the object
        :rtype: int
        """
        request_body = {
            ApiField.TAG_ID: tag_meta_id,
            ApiField.OBJECT_ID: object_id,
        }
        if value is not None:
            request_body[ApiField.VALUE] = value
        if frame_range is not None:
            request_body[ApiField.FRAME_RANGE] = frame_range

        response = self._api.post("annotation-objects.tags.add", request_body)
        id = response.json()[ApiField.ID]
        return id

    def remove(self, tag_id: int) -> None:
        """Remove tag from annotation object in point cloud.

        :param tag_id: tag ID of certain object
        :type tag_id: int
        """
        request_body = {ApiField.ID: tag_id}

        self._api.post("annotation-objects.tags.remove", request_body)

    def update(self, tag_id: int, value: Union[str, int]) -> None:
        """Update tag value for annotation object in point cloud.
        You could use only those values, which are correspond to TagMeta `value_type` and `possible_values`

        :param tag_id: tag ID of certain object
        :type tag_id: int
        :param value: possible_values from TagMeta
        :type value: Union[str, int]
        """
        request_body = {
            ApiField.ID: tag_id,
            ApiField.VALUE: value,
        }
        self._api.post("annotation-objects.tags.update-value", request_body)
