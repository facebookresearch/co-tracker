# coding: utf-8

from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.api.module_api import ApiField
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import (
    PointcloudEpisodeAnnotation,
)
from supervisely.video_annotation.key_id_map import KeyIdMap


class PointcloudEpisodeAnnotationAPI(EntityAnnotationAPI):
    """
    :class:`PointcloudEpisodeAnnotation<supervisely.pointcloud_annotation.pointcloud_episode_annotation.PointcloudEpisodeAnnotation>` for a point clouds episodes.
    :class:`PointcloudEpisodeAnnotationAPI<PointcloudEpisodeAnnotationAPI>` object is immutable.

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

        dataset_id = 62664
        ann_info = api.pointcloud_episode.annotation.download(dataset_id)
        print(ann_info)
    """

    _method_download = "point-clouds.episodes.annotations.info"
    _entity_ids_str = ApiField.POINTCLOUD_IDS

    def download(self, dataset_id: int) -> dict:
        """
        Download information about PointcloudEpisodeAnnotation by dataset ID from API.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :return: Dictionary woth information about PointcloudEpisodeAnnotation in json format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from pprint import pprint

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 62664
            ann_info = api.pointcloud_episode.annotation.download(dataset_id)
            pprint(ann_info)

            # Output: {
            #     'datasetId': 62664,
            #     'description': '',
            #     'frames': [
            #         {'figures': [{'classId': None,
            #                     'createdAt': '2023-04-05T08:55:52.445Z',
            #                     'description': '',
            #                     'geometry': {'dimensions': {'x': 2.3652234,
            #                                                 'y': 23.291742,
            #                                                 'z': 3.326648},
            #                                 'position': {'x': 86.29707472161449,
            #                                             'y': -14.472597682830635,
            #                                             'z': 0.8842007608554671},
            #                                 'rotation': {'x': 0,
            #                                             'y': 0,
            #                                             'z': -1.6962800995995606}},
            #                     'geometryType': 'cuboid_3d',
            #                     'id': 87830452,
            #                     'labelerLogin': 'almaz',
            #                     'objectId': 5565741,
            #                     'updatedAt': '2023-04-05T08:55:52.445Z'},
            #                     {'classId': None,
            #                     'createdAt': '2023-04-05T08:55:52.445Z',
            #                     'description': '',
            #                     'geometry': {'indices': [783,
            #                                             784,
            #                                             ...
            #                                             28326,
            #                                             30294]},
            #                     'geometryType': 'point_cloud',
            #                     'id': 87830456,
            #                     'labelerLogin': 'almaz',
            #                     'objectId': 5565740,
            #                     'updatedAt': '2023-04-05T08:55:52.445Z'}],
            #         'index': 0,
            #         'pointCloudId': 19618654},...
            #     ],
            #     'tags': []
            # }
        """

        response = self._api.post(self._method_download, {ApiField.DATASET_ID: dataset_id})
        return response.json()[0]

    def download_bulk(self, dataset_id, entity_ids):
        """Not supported for point cloud episodes."""
        raise RuntimeError("Not supported for episodes")

    def append(
        self,
        dataset_id: int,
        ann: PointcloudEpisodeAnnotation,
        frame_to_pointcloud_ids,
        key_id_map: KeyIdMap = None,
    ) -> None:
        """
        Loads an PointcloudEpisodeAnnotation to a given point cloud ID in the API.

        :param pointcloud_id: Point cloud ID in Supervisely.
        :type pointcloud_id: int
        :param ann: PointcloudEpisodeAnnotation object.
        :type ann: PointcloudEpisodeAnnotation
        :param frame_to_pointcloud_ids:  List of dictionaries with frame_id and name of pointcloud episodes
        :type frame_to_pointcloud_ids: List[dict]
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pcd_episodes_id = 198704259
            api.pointcloud_episode.annotation.append(pcd_episodes_id, pointcloud_ann)
        """

        if key_id_map is None:
            # create for internal purposes (to link figures and tags to objects)
            key_id_map = KeyIdMap()

        figures = []
        pointcloud_ids = []
        for i, frame in enumerate(ann.frames):
            for fig in frame.figures:
                if frame_to_pointcloud_ids.get(i) is None:  # skip unmapped frames
                    continue

                figures.append(fig)
                pointcloud_ids.append(frame_to_pointcloud_ids[i])

        if len(pointcloud_ids) == 0:
            return
        self._api.pointcloud_episode.object.append_bulk(pointcloud_ids[0], ann.objects, key_id_map)
        self._api.pointcloud_episode.figure.append_to_dataset(
            dataset_id, figures, pointcloud_ids, key_id_map
        )
