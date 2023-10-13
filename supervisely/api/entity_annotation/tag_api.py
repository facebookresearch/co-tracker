# coding: utf-8

from supervisely.api.module_api import ModuleApi
from supervisely.api.module_api import ApiField
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.video_annotation.key_id_map import KeyIdMap


class TagApi(ModuleApi):
    """"""

    _entity_id_field = None
    """"""
    _method_bulk_add = None
    """"""

    @staticmethod
    def info_sequence():
        """
        NamedTuple TagInfo information about Tag.

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            info_sequence = api.video.tag.info_sequence()
        """

        return [ApiField.ID,
                ApiField.PROJECT_ID,
                ApiField.NAME,
                ApiField.SETTINGS,
                ApiField.COLOR,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT
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

            tuple_name = api.video.tag.info_tuple_name()
            print(tuple_name) # TagInfo
        """

        return 'TagInfo'

    def get_list(self, project_id: int, filters=None):
        """
        Get list of tags for a given project ID.

        :param project_id: :class:`Dataset<supervisely.project.project.Project>` ID in Supervisely.
        :type project_id: int
        :param filters: List of parameters to sort output tags. See: https://dev.supervise.ly/api-docs/#tag/Advanced/paths/~1tags.list/get
        :type filters: List[Dict[str, str]], optional
        :return: List of the tags from the project with given id.
        :rtype: list
        """

        return self.get_list_all_pages('tags.list', {ApiField.PROJECT_ID: project_id, "filter": filters or []})

    def get_name_to_id_map(self, project_id: int):
        """
        Get dictionary with mapping tag name to tag ID for a given project ID.

        :param project_id: :class:`Dataset<supervisely.project.project.Project>` ID in Supervisely.
        :type project_id: int
        :return: Dictionary with mapping tag name to tag id for a given project ID.
        :rtype: dict
        """

        tags_info = self.get_list(project_id)
        return {tag_info.name: tag_info.id for tag_info in tags_info}

    def _tags_to_json(self, tags: KeyIndexedCollection, tag_name_id_map=None, project_id=None):
        """"""
        if tag_name_id_map is None and project_id is None:
            raise RuntimeError("Impossible to get ids for project tags")
        if tag_name_id_map is None:
            tag_name_id_map = self.get_name_to_id_map(project_id)
        tags_json = []
        tags_keys = []
        for tag in tags:
            tag_json = tag.to_json()
            tag_json[ApiField.TAG_ID] = tag_name_id_map[tag.name]
            tags_json.append(tag_json)
            tags_keys.append(tag.key())
        return tags_json, tags_keys

    def append_to_entity(self, entity_id: int, project_id: int, tags: KeyIndexedCollection, key_id_map: KeyIdMap = None):
        """
        Add tags to entity in project with given ID.

        :param entity_id: ID of the entity in Supervisely to add a tag to
        :type entity_id: int
        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param tags: Collection of tags
        :type tags: KeyIndexedCollection
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: List of tags IDs
        :rtype: list
        """

        if len(tags) == 0:
            return []
        tags_json, tags_keys = self._tags_to_json(tags, project_id=project_id)
        ids = self._append_json(entity_id, tags_json)
        KeyIdMap.add_tags_to(key_id_map, tags_keys, ids)
        return ids

    def _append_json(self, entity_id, tags_json):
        """"""
        if self._method_bulk_add is None:
            raise RuntimeError("self._method_bulk_add is not defined in child class")
        if self._entity_id_field is None:
            raise RuntimeError("self._entity_id_field is not defined in child class")

        if len(tags_json) == 0:
            return []
        response = self._api.post(self._method_bulk_add, {self._entity_id_field: entity_id, ApiField.TAGS: tags_json})
        ids = [obj[ApiField.ID] for obj in response.json()]
        return ids

    def append_to_objects(self, entity_id: int, project_id: int, objects: KeyIndexedCollection, key_id_map: KeyIdMap):
        """
        Add Tags to Annotation Objects.

        :param entity_id: ID of the entity in Supervisely to add a tag to
        :type entity_id: int
        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param tags_json: Collection of tags
        :type tags_json: KeyIndexedCollection
        :return: List of tags IDs
        :rtype: list
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pointcloud_id = 19373170
            pcd_info = api.
        """

        tag_name_id_map = self.get_name_to_id_map(project_id)

        tags_to_add = []
        tags_keys = []
        for object in objects:
            obj_id = key_id_map.get_object_id(object.key())
            if obj_id is None:
                raise RuntimeError("Can not add tags to object: OBJECT_ID not found for key {}".format(object.key()))
            tags_json, cur_tags_keys = self._tags_to_json(object.tags, tag_name_id_map=tag_name_id_map)
            for tag in tags_json:
                tag[ApiField.OBJECT_ID] = obj_id
                tags_to_add.append(tag)
            tags_keys.extend(cur_tags_keys)

        if len(tags_keys) != len(tags_to_add):
            raise RuntimeError("SDK error: len(tags_keys) != len(tags_to_add)")
        if len(tags_keys) == 0:
            return
        ids = self.append_to_objects_json(entity_id, tags_to_add)
        KeyIdMap.add_tags_to(key_id_map, tags_keys, ids)

    def append_to_objects_json(self, entity_id: int, tags_json: dict) -> list:
        """
        Add Tags to Annotation Objects.

        :param entity_id: ID of the entity in Supervisely to add a tag to
        :type entity_id: int
        :param tags_json: Collection of tags in JSON format
        :type tags_json: dict
        :return: List of tags IDs
        :rtype: list
        """

        if len(tags_json) == 0:
            return []
        response = self._api.post('annotation-objects.tags.bulk.add',
                                  {ApiField.ENTITY_ID: entity_id, ApiField.TAGS: tags_json})
        ids = [obj[ApiField.ID] for obj in response.json()]
        return ids
