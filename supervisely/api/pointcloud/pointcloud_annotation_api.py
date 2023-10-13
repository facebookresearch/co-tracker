# coding: utf-8

# docs
from typing import List, NamedTuple, Dict, Optional


from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI


class PointcloudAnnotationAPI(EntityAnnotationAPI):
    """
    :class:`PointcloudAnnotation<supervisely.pointcloud_annotation.pointcloud_annotation.PointcloudAnnotation>` for a single point cloud. :class:`PointcloudAnnotationAPI<PointcloudAnnotationAPI>` object is immutable.

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

        pointcloud_id = 19618685
        ann_info = api.pointcloud.annotation.download(src_pointcloud_id)
    """

    _method_download_bulk = "point-clouds.annotations.bulk.info"
    _entity_ids_str = ApiField.POINTCLOUD_IDS

    def download(self, pointcloud_id: int) -> List[Dict]:
        """
        Download information about PointcloudAnnotation by point cloud ID from API.

        :param pointcloud_id: Point cloud ID in Supervisely.
        :type pointcloud_id: int
        :return: Information about PointcloudAnnotation in json format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pointcloud_id = 19618685
            ann_info = api.pointcloud.annotation.download(src_pointcloud_id)
            print(ann_info)

            # Output: {
            #     'datasetId': 62664,
            #     'description': '',
            #     'frames': [{'figures': [{'classId': None,
            #                             'createdAt': '2023-04-05T08:55:52.526Z',
            #                             'description': '',
            #                             'geometry': {'dimensions': {'x': 1.6056261,
            #                                                         'y': 3.8312221,
            #                                                         'z': 1.8019634},
            #                                         'position': {'x': 10.181290418102629,
            #                                                     'y': 9.033275311626847,
            #                                                     'z': -0.9065660704238034},
            #                                         'rotation': {'x': 0,
            #                                                     'y': 0,
            #                                                     'z': 1.5985649758590998}},
            #                             'geometryType': 'cuboid_3d',
            #                             'id': 87830573,
            #                             'labelerLogin': 'almaz',
            #                             'objectId': 5565738,
            #                             'updatedAt': '2023-04-05T08:55:52.526Z'},
            #                             {'classId': None,
            #                             'createdAt': '2023-04-05T08:55:52.526Z',
            #                             'description': '',
            #                             'geometry': {'dimensions': {'x': 2.3652234,
            #                                                         'y': 23.291742,
            #                                                         'z': 3.326648},
            #                                         'position': {'x': 77.40255111910977,
            #                                                     'y': -9.582723835261527,
            #                                                     'z': 1.0131292020311293},
            #                                         'rotation': {'x': 0,
            #                                                     'y': 0,
            #                                                     'z': -1.5823898471886868}},
            #                             'geometryType': 'cuboid_3d',
            #                             'id': 87830574,
            #                             'labelerLogin': 'almaz',
            #                             'objectId': 5565741,
            #                             'updatedAt': '2023-04-05T08:55:52.526Z'}],
            #                 'index': 0,
            #                 'pointCloudId': 19618685}],
            #     'framesCount': 1,
            #     'objects': [{'classId': 683259,
            #                 'classTitle': 'Car',
            #                 'createdAt': '2023-04-05T08:55:52.384Z',
            #                 'datasetId': 62664,
            #                 'entityId': None,
            #                 'id': 5565737,
            #                 'labelerLogin': 'almaz',
            #                 'tags': [],
            #                 'updatedAt': '2023-04-05T08:55:52.384Z'}],
            #     'tags': []
            # }
        """

        info = self._api.pointcloud.get_info_by_id(pointcloud_id)
        return self._download(info.dataset_id, pointcloud_id)

    def append(
        self,
        pointcloud_id: int,
        ann: PointcloudAnnotation,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        """
        Loads an PointcloudAnnotation to a given point cloud ID in the API.

        :param pointcloud_id: Point cloud ID in Supervisely.
        :type pointcloud_id: int
        :param ann: PointcloudAnnotation object.
        :type ann: PointcloudAnnotation
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

            pointcloud_id = 198704259
            api.pointcloud.annotation.append(pointcloud_id, pointcloud_ann)
        """

        info = self._api.pointcloud.get_info_by_id(pointcloud_id)

        new_objects = []
        for object_3d in ann.objects:
            if key_id_map is not None and key_id_map.get_object_id(object_3d.key()) is not None:
                # object already uploaded
                continue
            new_objects.append(object_3d)

        self._append(
            self._api.pointcloud.tag,
            self._api.pointcloud.object,
            self._api.pointcloud.figure,
            info.project_id,
            info.dataset_id,
            pointcloud_id,
            ann.tags,
            PointcloudObjectCollection(new_objects),
            ann.figures,
            key_id_map,
        )
