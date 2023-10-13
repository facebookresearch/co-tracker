# coding: utf-8
from __future__ import annotations
from typing import List, NamedTuple, Dict, Optional

from supervisely.api.module_api import ApiField, ModuleApi, RemoveableBulkModuleApi
from supervisely.video_annotation.key_id_map import KeyIdMap


class ObjectApi(RemoveableBulkModuleApi):
    """
    Object for :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`.
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple ObjectInfo information about Object.

        :Example:

         .. code-block:: python

            ObjectInfo(id=152118,
                       description='',
                       created_at='2021-03-23T13:25:34.705Z',
                       updated_at='2021-03-23T13:25:34.705Z',
                       dataset_id=466642, class_id=2856942,
                       entity_id=198703211,
                       tags=[{'objectId': 152118, 'tagId': 29098694, 'entityId': None, 'id': 40632, 'value': 'grey'}],
                       meta={},
                       created_by_id=16154)
        """
        return [
            ApiField.ID,
            ApiField.DESCRIPTION,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.DATASET_ID,
            ApiField.CLASS_ID,
            ApiField.ENTITY_ID,
            ApiField.TAGS,
            ApiField.META,
            ApiField.CREATED_BY_ID,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of NamedTuple for class.

        :return: NamedTuple name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            tuple_name = api.video.object.info_tuple_name()
            print(tuple_name) # ObjectInfo
        """

        return "ObjectInfo"

    def get_info_by_id(self, id: int) -> NamedTuple:
        """
        Get Object information by ID.

        :param id: Object ID in Supervisely.
        :type id: int
        :return: Information about Object. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`NamedTuple`
        """
        return self._get_info_by_id(id, "annotation-objects.info")

    def get_list(
        self, dataset_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[NamedTuple]:
        """
        Get list of information about all video Objects for a given dataset ID.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Objects.
        :type filters: List[dict], optional
        :return: Information about Objects. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 466642
            object_infos = api.video.object.get_list(dataset_id)
            print(object_infos)
            # Output: [
            #     [
            #         152118,
            #         "",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         466642,
            #         2856942,
            #         198703211,
            #         [
            #             {
            #                 "objectId": 152118,
            #                 "tagId": 29098694,
            #                 "entityId": null,
            #                 "id": 40632,
            #                 "value": "grey"
            #             }
            #         ],
            #         {},
            #         16154
            #     ],
            #     [
            #         152119,
            #         "",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         466642,
            #         2856942,
            #         198703211,
            #         [
            #             {
            #                 "objectId": 152119,
            #                 "tagId": 29098694,
            #                 "entityId": null,
            #                 "id": 40633,
            #                 "value": "wine"
            #             }
            #         ],
            #         {},
            #         16154
            #     ],
            #     [
            #         152120,
            #         "",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         466642,
            #         2856942,
            #         198703211,
            #         [
            #             {
            #                 "objectId": 152120,
            #                 "tagId": 29098694,
            #                 "entityId": null,
            #                 "id": 40634,
            #                 "value": "beige"
            #             }
            #         ],
            #         {},
            #         16154
            #     ],
            #     [
            #         152121,
            #         "",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         466642,
            #         2856941,
            #         198703212,
            #         [
            #             {
            #                 "objectId": 152121,
            #                 "tagId": 29098696,
            #                 "entityId": null,
            #                 "id": 40635,
            #                 "value": "juvenile"
            #             }
            #         ],
            #         {},
            #         16154
            #     ],
            #     [
            #         152122,
            #         "",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         466642,
            #         2856943,
            #         198703211,
            #         [],
            #         {},
            #         16154
            #     ]
            # ]
        """
        return self.get_list_all_pages(
            "annotation-objects.list",
            {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters or []},
        )

    def _append_bulk(
        self,
        tag_api,
        entity_id,
        project_id,
        dataset_id,
        objects,
        key_id_map: KeyIdMap = None,
        is_pointcloud=False,
    ):
        """"""
        if len(objects) == 0:
            return []

        objcls_name_id_map = self._api.object_class.get_name_to_id_map(project_id)

        items = []
        for obj in objects:
            new_obj = {ApiField.CLASS_ID: objcls_name_id_map[obj.obj_class.name]}

            if not is_pointcloud:
                # if entity_id is not None:
                new_obj[ApiField.ENTITY_ID] = entity_id
            items.append(new_obj)

        response = self._api.post(
            "annotation-objects.bulk.add",
            {ApiField.DATASET_ID: dataset_id, ApiField.ANNOTATION_OBJECTS: items},
        )
        ids = [obj[ApiField.ID] for obj in response.json()]
        KeyIdMap.add_objects_to(key_id_map, [obj.key() for obj in objects], ids)

        # add tags to objects
        tag_api.append_to_objects(entity_id, project_id, objects, key_id_map)

        return ids
