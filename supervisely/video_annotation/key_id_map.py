# coding: utf-8
from __future__ import annotations
from typing import List, Dict, Optional
import uuid
from uuid import UUID
from bidict import bidict
from supervisely.io.json import dump_json_file, load_json_file

TAGS = "tags"
OBJECTS = "objects"
FIGURES = "figures"
VIDEOS = "videos"

ALLOWED_KEY_TYPES = [TAGS, OBJECTS, VIDEOS, FIGURES]

# @TODO: reimplement to support different item types - videos, volumes, 3d episodes, ...
class KeyIdMap:
    """
    KeyIdMap object for :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`. It consist from dict with bidict values.

    :Usage example:

     .. code-block:: python

        key_id_map = KeyIdMap()
        print(key_id_map.to_dict())
        # Output: {
        #     "tags": {},
        #     "objects": {},
        #     "figures": {},
        #     "videos": {}
        # }
    """

    def __init__(self):
        self._data = dict()
        self._data[TAGS] = bidict()
        self._data[OBJECTS] = bidict()
        self._data[FIGURES] = bidict()
        self._data[VIDEOS] = bidict()

    def _add(self, key_type, key: uuid.UUID, id: Optional[int] = None):
        """
        Add given data in self._data dictionary. Raise error if data type of any parameter is invalid
        :param key_type: str
        :param key: uuid class object
        :param id: int
        """
        if key_type not in ALLOWED_KEY_TYPES:
            raise RuntimeError(
                "Key type {!r} is not allowed. Allowed types are {}".format(
                    key_type, ALLOWED_KEY_TYPES
                )
            )
        if type(key) is not uuid.UUID:
            raise RuntimeError("Key should be of type uuid.UUID")
        if id is not None and type(id) is not int:
            raise RuntimeError("Id should be of type int")
        self._data[key_type].update(bidict({key: id}))

    def add_object(self, key: UUID, id: int) -> None:
        """
        Add UUID and ID of :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` in KeyIdMap.

        :param key: UUID object.
        :type key: UUID
        :param id: :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` ID.
        :type id: int
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            new_uuid = uuid.uuid4() # "0c0033c5b4834d4cbabece4317295f07"
            key_id_map.add_object(new_uuid, 1)
            print(key_id_map.to_dict())
            # Output: {
            #     "tags": {},
            #     "objects": {
            #         "0c0033c5b4834d4cbabece4317295f07": 1
            #     },
            #     "figures": {},
            #     "videos": {}
            # }
        """
        self._add(OBJECTS, key, id)

    def add_tag(self, key: UUID, id: int) -> None:
        """
        Add UUID and ID of :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` in KeyIdMap.

        :param key: UUID object.
        :type key: UUID
        :param id: :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` ID.
        :type id: int
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            new_uuid = uuid.uuid4() # "697d005df2a94bb386188c78a61b0a86"
            key_id_map.add_tag(new_uuid, 34)
            print(key_id_map.to_dict())
            # Output: {
            #     "tags": {
            #         "697d005df2a94bb386188c78a61b0a86": 34
            #     },
            #     "objects": {},
            #     "figures": {},
            #     "videos": {}
            # }
        """
        self._add(TAGS, key, id)

    def add_figure(self, key: UUID, id: int) -> None:
        """
        Add UUID and ID of :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` in KeyIdMap.

        :param key: UUID object.
        :type key: UUID
        :param id: :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` ID.
        :type id: int
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            new_uuid = uuid.uuid4() # "ac1018e6673d405590086063af8184ca"
            key_id_map.add_figure(new_uuid, 55)
            print(key_id_map.to_dict())
            # Output: {
            #     "tags": {},
            #     "objects": {},
            #     "figures": {
            #         "ac1018e6673d405590086063af8184ca": 55
            #     },
            #     "videos": {}
            # }
        """
        self._add(FIGURES, key, id)

    def add_video(self, key: UUID, id: int) -> None:
        """
        Add UUID and ID of :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` in KeyIdMap.

        :param key: UUID object.
        :type key: UUID
        :param id: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` ID.
        :type id: int
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            new_uuid = uuid.uuid4() # "775f2c581cec44ca8c10419c20c52fcc"
            key_id_map.add_video(new_uuid, 77)
            print(key_id_map.to_dict())
            # Output: {
            #     "tags": {},
            #     "objects": {},
            #     "figures": {},
            #     "videos": {
            #         "775f2c581cec44ca8c10419c20c52fcc": 77
            #     }
            # }
        """
        self._add(VIDEOS, key, id)

    def _get_id_by_key(self, key_type, key: uuid.UUID):
        """
        :param key_type: str
        :param key: uuid class object
        :return: Id by given key. None if there is no such key. Raise error if key type is not uuid.UUID
        """
        if type(key) is not uuid.UUID:
            raise RuntimeError("Key should be of type uuid.UUID")

        if key in self._data[key_type]:
            return self._data[key_type][key]
        else:
            return None

    def _get_key_by_id(self, key_type, id: int):
        """
        :param key_type: str
        :param id: int
        :return: Key by given id. None if there is no such id. Raise error if id type is not int
        """
        if type(id) is not int:
            raise RuntimeError("Id should be of type int")
        if id not in self._data[key_type].inverse:
            return None
        return self._data[key_type].inverse[id]

    def get_object_id(self, key: UUID) -> int:
        """
        Get :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` ID.

        :param key: UUID object.
        :type key: UUID
        :return: :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` ID
        :rtype: :class:`int`

        :Usage example:

         .. code-block:: python

            obj_uuid = '0c0033c5b4834d4cbabece4317295f07'
            obj_id = key_id_map.get_object_id(obj_uuid) # 1
        """
        return self._get_id_by_key(OBJECTS, key)

    def get_tag_id(self, key: UUID) -> int:
        """
        Get :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` ID.

        :param key: UUID object.
        :type key: UUID
        :return: :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` ID
        :rtype: :class:`int`

        :Usage example:

         .. code-block:: python

            tag_uuid = '697d005df2a94bb386188c78a61b0a86'
            tag_id = key_id_map.get_tag_id(tag_uuid) # 34
        """
        return self._get_id_by_key(TAGS, key)

    def get_figure_id(self, key: UUID) -> int:
        """
        Get :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` ID.

        :param key: UUID object.
        :type key: UUID
        :return: :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` ID
        :rtype: :class:`int`

        :Usage example:

         .. code-block:: python

            figure_uuid = 'ac1018e6673d405590086063af8184ca'
            figure_id = key_id_map.get_figure_id(figure_uuid) # 55
        """
        return self._get_id_by_key(FIGURES, key)

    def get_video_id(self, key: UUID) -> int:
        """
        Get :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` ID.

        :param key: UUID object.
        :type key: UUID
        :return: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` ID
        :rtype: :class:`int`

        :Usage example:

         .. code-block:: python

            video_uuid = '775f2c581cec44ca8c10419c20c52fcc'
            video_id = key_id_map.get_video_id(video_uuid) # 77
        """
        return self._get_id_by_key(VIDEOS, key)

    def get_object_key(self, id: int) -> UUID:
        """
        Get :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` UUID key.

        :param key: :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` ID.
        :type key: int
        :return: :class:`UUID` object
        :rtype: :class:`UUID`

        :Usage example:

         .. code-block:: python

            obj_id = 1
            obj_uuid = key_id_map.get_object_id(obj_id) # '0c0033c5b4834d4cbabece4317295f07'
        """
        return self._get_key_by_id(OBJECTS, id)

    def get_tag_key(self, id: int) -> UUID:
        """
        Get :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` UUID key.

        :param key: :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` ID.
        :type key: int
        :return: :class:`UUID` object
        :rtype: :class:`UUID`

        :Usage example:

         .. code-block:: python

            tag_id = 34
            tag_uuid = key_id_map.get_tag_key(tag_id) # '697d005df2a94bb386188c78a61b0a86'
        """
        return self._get_key_by_id(TAGS, id)

    def get_figure_key(self, id: int) -> UUID:
        """
        Get :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` UUID key.

        :param key: :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` ID.
        :type key: int
        :return: :class:`UUID` object
        :rtype: :class:`UUID`

        :Usage example:

         .. code-block:: python

            figure_id = 55
            figure_uuid = key_id_map.get_figure_key(figure_id) # 'ac1018e6673d405590086063af8184ca'
        """
        return self._get_key_by_id(FIGURES, id)

    def get_video_key(self, id: int) -> UUID:
        """
        Get :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` UUID key.

        :param key: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` ID.
        :type key: int
        :return: :class:`UUID` object
        :rtype: :class:`UUID`

        :Usage example:

         .. code-block:: python

            video_id = 77
            video_uuid = key_id_map.get_video_key(video_id) # '775f2c581cec44ca8c10419c20c52fcc'
        """
        return self._get_key_by_id(VIDEOS, id)

    def to_dict(self) -> Dict[str, Dict]:
        """
        Convert the KeyIdMap to a dict(bidict values to dictionary with dict values).

        :return: Json format as a dict
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            print(key_id_map.to_dict())
            # Output: {
            #     "tags": {},
            #     "objects": {},
            #     "figures": {},
            #     "videos": {}
            # }
        """
        simple_dict = {}
        for type_str, value_bidict in self._data.items():
            sub_dict = {}
            for uuid, int_id in value_bidict.items():
                sub_dict[uuid.hex] = int_id
            simple_dict[type_str] = sub_dict
        return simple_dict

    def dump_json(self, path: str) -> None:
        """
        Write KeyIdMap to file with given path.

        :param path: Target file path.
        :type path: str
        :return: :class:`None`
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            key_id_map.dump_json('/home/admin/work/projects/key_id.json')
        """
        simple_dict = self.to_dict()
        dump_json_file(simple_dict, path, indent=4)

    @classmethod
    def load_json(cls, path: str) -> KeyIdMap:
        """
        Decoding data from json file with given filename to KeyIdMap.

        :param path: Target file path.
        :type path: str
        :return: KeyIdMap object
        :rtype: :class:`KeyIdMap`

        :Usage example:

         .. code-block:: python

            new_key_id = KeyIdMap.load_json('/home/admin/work/projects/key_id.json')
        """
        simple_dict = load_json_file(path)
        result = cls()
        for key_type, value_dict in simple_dict.items():
            for key_str, id in value_dict.items():
                result._add(key_type, uuid.UUID(key_str), id)
        return result

    @classmethod
    def _add_to(cls, key_id_map, key_type, keys, ids):
        """
        Add given values(keys, ids) to KeyIdMap class object with given type of key
        :param key_id_map: KeyIdMap class object
        :param key_type: str
        :param keys: list of uuid class objects
        :param ids: list of integers
        :return: None if key_id_map parameter is None
        """
        if key_id_map is None:
            return
        for key, id in zip(keys, ids):
            key_id_map._add(key_type, key, id)

    @classmethod
    def add_tags_to(cls, key_id_map: KeyIdMap, keys: List[UUID], ids: List[int]) -> None:
        """
        Add :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` type of instances with given values(keys, ids) to KeyIdMap object.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param key: List of UUID objects.
        :type key: List[UUID]
        :param id: List of :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` IDs.
        :type id: List[int]
        :return: :class:`None`
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            uuid_1 = uuid.uuid4()
            tag_id_1 = 1213
            uuid_2 = uuid.uuid4()
            tag_id_2 = 3686
            KeyIdMap.add_tags_to(key_id_map, [uuid_1, uuid_2], [tag_id_1, tag_id_2])
        """
        cls._add_to(key_id_map, TAGS, keys, ids)

    @classmethod
    def add_tag_to(cls, key_id_map: KeyIdMap, key: UUID, id: int) -> None:
        """
        Add :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` type of instance with given key and id to KeyIdMap object.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param key: UUID object.
        :type key: UUID
        :param id: :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` ID.
        :type id: int
        :return: :class:`None`
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            new_uuid = uuid.uuid4()
            new_tag_id = 1213
            KeyIdMap.add_tag_to(key_id_map, new_uuid, new_tag_id)
        """
        cls._add_tags_to(key_id_map, [key], [id])

    @classmethod
    def add_objects_to(cls, key_id_map: KeyIdMap, keys: List[UUID], ids: List[int]) -> None:
        """
        Add :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` type of instances with given values(keys, ids) to KeyIdMap object.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param key: List of UUID objects.
        :type key: List[UUID]
        :param id: List of :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` IDs.
        :type id: List[int]
        :return: :class:`None`
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            uuid_1 = uuid.uuid4()
            object_id_1 = 23
            uuid_2 = uuid.uuid4()
            object_id_2 = 57
            KeyIdMap.add_objects_to(key_id_map, [uuid_1, uuid_2], [object_id_1, object_id_2])
        """
        cls._add_to(key_id_map, OBJECTS, keys, ids)

    @classmethod
    def add_object_to(cls, key_id_map: KeyIdMap, key: UUID, id: int) -> None:
        """
        Add :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` type of instance with given key and id to KeyIdMap object.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param key: UUID object.
        :type key: UUID
        :param id: :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` ID.
        :type id: int
        :return: :class:`None`
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            new_uuid = uuid.uuid4()
            new_object_id = 76
            KeyIdMap.add_object_to(key_id_map, new_uuid, new_object_id)
        """
        cls._add_objects_to(key_id_map, [key], [id])

    @classmethod
    def add_figures_to(cls, key_id_map: KeyIdMap, keys: List[UUID], ids: List[int]) -> None:
        """
        Add :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` type of instances with given values(keys, ids) to KeyIdMap object.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param key: List of UUID objects.
        :type key: List[UUID]
        :param id: List of :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` IDs.
        :type id: List[int]
        :return: :class:`None`
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            uuid_1 = uuid.uuid4()
            figure_id_1 = 23
            uuid_2 = uuid.uuid4()
            figure_id_2 = 57
            KeyIdMap.add_figures_to(key_id_map, [uuid_1, uuid_2], [figure_id_1, figure_id_2])
        """
        cls._add_to(key_id_map, FIGURES, keys, ids)

    @classmethod
    def add_figure_to(cls, key_id_map: KeyIdMap, key: UUID, id: int) -> None:
        """
        Add :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` type of instance with given key and id to KeyIdMap object.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param key: UUID object.
        :type key: UUID
        :param id: :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` ID.
        :type id: int
        :return: :class:`None`
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            new_uuid = uuid.uuid4()
            new_figure_id = 3834
            KeyIdMap.add_figure_to(key_id_map, new_uuid, new_figure_id)
        """
        cls._add_figures_to(key_id_map, [key], [id])

    @classmethod
    def add_videos_to(cls, key_id_map: KeyIdMap, keys: List[UUID], ids: List[int]) -> None:
        """
        Add :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` type of instances with given values(keys, ids) to KeyIdMap object.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param key: List of UUID objects.
        :type key: List[UUID]
        :param id: List of :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` IDs.
        :type id: List[int]
        :return: :class:`None`
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            uuid_1 = uuid.uuid4()
            video_id_1 = 567
            uuid_2 = uuid.uuid4()
            video_id_2 = 5200
            KeyIdMap.add_videos_to(key_id_map, [uuid_1, uuid_2], [video_id_1, video_id_2])
        """
        cls._add_to(key_id_map, VIDEOS, keys, ids)

    @classmethod
    def add_video_to(cls, key_id_map: KeyIdMap, key: UUID, id: int) -> None:
        """
        Add :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` type of instance with given key and id to KeyIdMap object.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param key: UUID object.
        :type key: UUID
        :param id: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` ID.
        :type id: int
        :return: :class:`None`
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            key_id_map = KeyIdMap()
            new_uuid = uuid.uuid4()
            new_video_id = 3834
            KeyIdMap.add_video_to(key_id_map, new_uuid, new_video_id)
        """
        cls._add_videos_to(key_id_map, [key], [id])
