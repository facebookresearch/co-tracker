# coding: utf-8

from supervisely.api.pointcloud.pointcloud_object_api import PointcloudObjectApi
from supervisely.api.pointcloud.pointcloud_episode_tag_api import PointcloudEpisodeObjectTagApi


class PointcloudEpisodeObjectApi(PointcloudObjectApi):
    """
    :class:`PointcloudEpisodeObject<supervisely.pointcloud_annotation.pointcloud_object.PointcloudEpisodeObject>` for :class:`PointcloudEpisodeAnnotation<supervisely.pointcloud_annotation.pointcloud_episode_annotation.PointcloudEpisodeAnnotation>`.
    """

    def __init__(self, api):
        """
        :param api: Api class object
        """
        super().__init__(api)
        self.tag = PointcloudEpisodeObjectTagApi(api)
