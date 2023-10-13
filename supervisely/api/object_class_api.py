# coding: utf-8
"""get list of :class:`objects<supervisely.annotation.obj_class.ObjClass>` from supervisely project"""

from __future__ import annotations
from typing import NamedTuple, List, Dict, Optional

from supervisely.api.module_api import ModuleApi
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap


class ObjectClassApi(ModuleApi):
    """
    API for working with :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>`. :class:`ObjectClassApi<ObjectClassApi>` object is immutable.

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

        project_id = 1951
        obj_class_infos = api.object_class.get_list(project_id)
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple ObjectClassInfo information about ObjectClass.

        :Example:

         .. code-block:: python

            ObjectClassInfo(id=22309,
                            name='lemon',
                            description='',
                            shape='bitmap',
                            color='#51C6AA',
                            settings={},
                            created_at='2021-03-02T10:04:33.973Z',
                            updated_at='2021-03-11T09:37:07.111Z')
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.DESCRIPTION,
            ApiField.SHAPE,
            ApiField.COLOR,
            ApiField.SETTINGS,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **ObjectClassInfo**.
        """
        return "ObjectClassInfo"

    def get_list(
        self, project_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[NamedTuple]:
        """
        List of ObjClasses in the given Project.

        :param project_id: Project ID in which the ObjClasses are located.
        :type project_id: int
        :param filters: List of params to sort output ObjClasses.
        :type filters: List[dict], optional
        :return: List of ObjClasses with information from the given Project. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 1951
            obj_class_infos = api.object_class.get_list(project_id)
            print(obj_class_infos)
            # Output: [ObjectClassInfo(id=22309,
            #                          name='lemon',
            #                          description='',
            #                          shape='bitmap',
            #                          color='#51C6AA',
            #                          settings={},
            #                          created_at='2021-03-02T10:04:33.973Z',
            #                          updated_at='2021-03-11T09:37:07.111Z'),
            #  ObjectClassInfo(id=22310,
            #                  name='kiwi',
            #                  description='',
            #                  shape='bitmap',
            #                  color='#FF0000',
            #                  settings={},
            #                  created_at='2021-03-02T10:04:33.973Z',
            #                  updated_at='2021-03-11T09:37:07.111Z')
            # ]

            obj_class_list = api.object_class.get_list(1951, filters=[{'field': 'name', 'operator': '=', 'value': 'lemon' }])
            print(obj_class_list)
            # Output: [
            #     [
            #         22309,
            #         "lemon",
            #         "",
            #         "bitmap",
            #         "#51C6AA",
            #         {},
            #         "2021-03-02T10:04:33.973Z",
            #         "2021-03-11T09:37:07.111Z"
            #     ]
            # ]
        """
        return self.get_list_all_pages(
            "advanced.object_classes.list",
            {ApiField.PROJECT_ID: project_id, "filter": filters or []},
        )

    def get_name_to_id_map(self, project_id: int) -> Dict[str, int]:
        """
        :param project_id: Project ID in which the ObjClasses are located.
        :type project_id: int
        :return: Dictionary Key ID Map {'key': id}
        :rtype: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            obj_class_map = api.object_class.get_name_to_id_map(1951)
            print(obj_class_map)
            # Output: {'lemon': 22309, 'kiwi': 22310, 'cucumber': 22379}
        """
        objects_infos = self.get_list(project_id)
        return {object_info.name: object_info.id for object_info in objects_infos}

    def _get_info_by_id(self, id, method, fields=None):
        response = self._get_response_by_id(id, method, id_field=ApiField.ID, fields=fields)
        return (
            self._convert_json_info(response.json(), skip_missing=True)
            if (response is not None)
            else None
        )

    def get_info_by_id(self, id):
        return self._get_info_by_id(
            id,
            "advanced.object_classes.info",
        )

    # def _object_classes_to_json(self, object_classes: KeyIndexedCollection, objclasses_name_id_map=None, project_id=None):
    #     pass #@TODO: implement
    #     # if objclasses_name_id_map is None and project_id is None:
    #     #     raise RuntimeError("Impossible to get ids for projectTags")
    #     # if objclasses_name_id_map is None:
    #     #     objclasses_name_id_map = self.get_name_to_id_map(project_id)
    #     # tags_json = []
    #     # for tag in tags:
    #     #     tag_json = tag.to_json()
    #     #     tag_json[ApiField.TAG_ID] = tag_name_id_map[tag.name]
    #     #     tags_json.append(tag_json)
    #     # return tags_json
    #
    # def append_to_video(self, video_id, tags: KeyIndexedCollection, key_id_map: KeyIdMap = None):
    #     if len(tags) == 0:
    #         return []
    #     video_info = self._api.video.get_info_by_id(video_id)
    #     tags_json = self._tags_to_json(tags, project_id=video_info.project_id)
    #     ids = self.append_to_video_json(video_id, tags_json)
    #     KeyIdMap.add_tags_to(key_id_map, [tag.key() for tag in tags], ids)
    #     return ids
    #
    # def append_to_video_json(self, video_id, tags_json):
    #     if len(tags_json) == 0:
    #         return
    #     response = self._api.post('videos.tags.bulk.add', {ApiField.VIDEO_ID: video_id, ApiField.TAGS: tags_json})
    #     ids = [obj[ApiField.ID] for obj in response.json()]
    #     return ids
