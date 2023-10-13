# coding: utf-8

# docs
from typing import Dict

from supervisely.api.module_api import ApiField
from supervisely.api.pointcloud.pointcloud_api import PointcloudApi
from supervisely.api.pointcloud.pointcloud_episode_annotation_api import (
    PointcloudEpisodeAnnotationAPI,
)
from supervisely.api.pointcloud.pointcloud_episode_object_api import PointcloudEpisodeObjectApi


class PointcloudEpisodeApi(PointcloudApi):
    """

    API for working with :class:`PointcloudEpisodes<supervisely.pointcloud_episodes.pointcloud_episodes>`.
    :class:`PointcloudEpisodeApi<PointcloudEpisodeApi>` object is immutable.
    Inherits from :class:`PointcloudApi<supervisely.api.pointcloud.PointcloudApi>`.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        pcd_epsodes_id = 19373295
        pcd_epsodes_info = api.pointcloud_episode.get_info_by_id(pcd_epsodes_id) # api usage example
    """

    def __init__(self, api):
        super().__init__(api)
        self.annotation = PointcloudEpisodeAnnotationAPI(api)
        self.object = PointcloudEpisodeObjectApi(api)
        self.tag = None

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.meta is not None:
            return res._replace(frame=res.meta[ApiField.FRAME])
        else:
            raise RuntimeError(
                "Error with point cloud meta or API version. Please, contact support"
            )

    def get_frame_name_map(self, dataset_id: int) -> Dict:
        """
        Get a dictionary with frame_id and name of pointcloud by dataset id.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :return: Dict with frame_id and name of pointcloud.
        :rtype: Dict

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 62664
            frame_to_name_map = api.pointcloud_episode.get_frame_name_map(dataset_id)
            print(frame_to_name_map)

            # Output:
            # {0: '001', 1: '002'}
        """

        pointclouds = self.get_list(dataset_id)

        frame_index_to_pcl_name = {}
        if len(pointclouds) > 0 and pointclouds[0].frame is None:
            pointclouds_names = sorted([x.name for x in pointclouds])
            for frame_index, pcl_name in enumerate(pointclouds_names):
                frame_index_to_pcl_name[frame_index] = pcl_name

        else:
            frame_index_to_pcl_name = {x.frame: x.name for x in pointclouds}

        return frame_index_to_pcl_name

    def notify_progress(
        self,
        track_id: int,
        dataset_id: int,
        pcd_ids: list,
        current: int,
        total: int,
    ):
        """
        Send message to the Annotation Tool and return info if tracking was stopped

        :param track_id: int
        :param dataset_id: int
        :param pcd_ids: list
        :param current: int
        :param total: int
        :return: str
        """

        response = self._api.post(
            "point-clouds.episodes.notify-annotation-tool",
            {
                "type": "point-cloud-episodes:fetch-figures-in-range",
                "data": {
                    ApiField.TRACK_ID: track_id,
                    ApiField.DATASET_ID: dataset_id,
                    ApiField.POINTCLOUD_IDS: pcd_ids,
                    ApiField.PROGRESS: {ApiField.CURRENT: current, ApiField.TOTAL: total},
                },
            },
        )
        return response.json()[ApiField.STOPPED]
