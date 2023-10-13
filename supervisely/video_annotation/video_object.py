# coding: utf-8

from __future__ import annotations
from typing import List, Dict, Optional
import uuid
from bidict import bidict

from supervisely.annotation.label import LabelJsonFields
from supervisely.annotation.obj_class import ObjClass
from supervisely.project.project_meta import ProjectMeta
from supervisely._utils import take_with_default
from supervisely.video_annotation.constants import KEY, ID, OBJECTS_MAP
from supervisely.video_annotation.video_tag_collection import VideoTagCollection
from supervisely.video_annotation.video_tag import VideoTag
from supervisely.collection.key_indexed_collection import KeyObject
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.geometry.constants import (
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    CLASS_ID,
)


class VideoObject(KeyObject):
    """
    VideoObject object for :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`. :class:`VideoObject<VideoObject>` object is immutable.

    :param obj_class: VideoObject :class:`class<supervisely.annotation.obj_class.ObjClass>`.
    :type obj_class: ObjClass
    :param tags: VideoObject :class:`tags<supervisely.video_annotation.video_tag_collection.VideoTagCollection>`.
    :type tags: VideoTagCollection, optional
    :param key: KeyIdMap object.
    :type key: KeyIdMap, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which VideoObject belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created VideoObject.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when VideoObject was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when VideoObject was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        obj_class_car = sly.ObjClass('car', sly.Rectangle)
        video_obj_car = sly.VideoObject(obj_class_car)
        video_obj_car_json = video_obj_car.to_json()
        print(video_obj_car_json)
        # Output: {
        #     "key": "6b819f1840f84d669b32cdec225385f0",
        #     "classTitle": "car",
        #     "tags": []
        # }
    """
    def __init__(self, obj_class: ObjClass, tags: Optional[VideoTagCollection] = None, key: Optional[uuid.UUID]=None,
                 class_id: Optional[int]=None, labeler_login: Optional[str]=None, updated_at: Optional[str]=None,
                 created_at: Optional[str]=None):
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at

        self._obj_class = obj_class
        self._key = take_with_default(key, uuid.uuid4())
        self._tags = take_with_default(tags, VideoTagCollection())
        self._class_id = take_with_default(class_id, None)

    def _add_creation_info(self, d):
        if self.labeler_login is not None:
            d[LABELER_LOGIN] = self.labeler_login
        if self.updated_at is not None:
            d[UPDATED_AT] = self.updated_at
        if self.created_at is not None:
            d[CREATED_AT] = self.created_at

    @property
    def obj_class(self) -> ObjClass:
        """
        ObjClass of the current VideoObject.

        :return: ObjClass object
        :rtype: :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            obj_car_json = video_obj_car.obj_class.to_json()
            print(obj_car_json)
            # Output: {
            #     "title": "car",
            #     "shape": "rectangle",
            #     "color": "#8A0F7B",
            #     "geometry_config": {},
            #     "hotkey": ""
            # }
        """
        return self._obj_class

    def key(self) -> uuid.UUID:
        """
        Object key.

        :return: Object key
        :rtype: uuid.UUID
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            key = video_obj_car.key()
            print(key)
            # Output: 158e6cf4f4ac4c639fc6994aad127c16
        """

        return self._key

    @property
    def tags(self) -> VideoTagCollection:
        """
        VideoTagCollection of the current VideoObject.

        :return: VideoTagCollection object
        :rtype: :class:`VideoTagCollection<supervisely.video_annotation.video_tag_collection.VideoTagCollection>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Create VideoTagCollection
            meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
            from supervisely.video_annotation.video_tag import VideoTag
            car_tag = VideoTag(meta_car, value='acura')
            from supervisely.video_annotation.video_tag_collection import VideoTagCollection
            video_tags = VideoTagCollection([car_tag])

            # Create VideoObject
            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car, video_tags)

            # Get VideoTagCollection
            tags = video_obj_car.tags
            tags_json = tags.to_json()
            print(tags_json)
            # Output: [
            #     {
            #         "name": "car_tag",
            #         "value": "acura",
            #         "key": "4f82fbcab74c44259d7a0e29d604602e"
            #     }
            # ]
        """
        return self._tags.clone()

    @property
    def class_id(self) -> int:
        """
        Object class ID.

        :return: Object class ID.
        :rtype: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            class_id = video_obj_car.class_id
        """

        return self._class_id

    def add_tag(self, tag: VideoTag) -> VideoObject:
        """
        Adds VideoTag to the current VideoObject.

        :param tag: VideoTag to be added.
        :type tag: VideoTag
        :return: VideoObject object
        :rtype: :class:`VideoObject<VideoObject>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Create VideoObject
            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)

            # Create VideoTag
            meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
            from supervisely.video_annotation.video_tag import VideoTag
            car_tag = VideoTag(meta_car, value='acura')

            # Add VideoTag
            new_obj_car = video_obj_car.add_tag(car_tag)
            new_obj_car_json = new_obj_car.to_json()
            print(new_obj_car_json)
            # Output: {
            #     "key": "1ab52285ee634c93b724fa655b785eae",
            #     "classTitle": "car",
            #     "tags": [
            #         {
            #             "name": "car_tag",
            #             "value": "acura",
            #             "key": "d9e52b275e074c538f162a6d679aed9e"
            #         }
            #     ]
            # }
        """
        return self.clone(tags=self._tags.add(tag))

    def add_tags(self, tags: List[VideoTag]) -> VideoObject:
        """
        Adds VideoTags to the current VideoObject.

        :param tag: List of VideoTags to be added.
        :type tag: List[VideoTag]
        :return: VideoObject object
        :rtype: :class:`VideoObject<VideoObject>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Create VideoObject
            obj_class_vehicle = sly.ObjClass('vehicle', sly.Rectangle)
            video_obj_vehicle = sly.VideoObject(obj_class_vehicle)

            # Create VideoTags
            from supervisely.video_annotation.video_tag import VideoTag
            meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
            car_tag = VideoTag(meta_car, value='acura')
            meta_bus = sly.TagMeta('bus_tag', sly.TagValueType.ANY_STRING)
            bus_tag = VideoTag(meta_bus, value='volvo')

            # Add VideoTags
            new_obj_vehicle = video_obj_vehicle.add_tags([car_tag, bus_tag])
            new_obj_vehicle_json = new_obj_vehicle.to_json()
            print(new_obj_vehicle_json)
            # Output: {
            #     "key": "94055c5e8cb146368f627fc608fb6b44",
            #     "classTitle": "vehicle",
            #     "tags": [
            #         {
            #             "name": "car_tag",
            #             "value": "acura",
            #             "key": "6679f47c96734565919fbffc278532a1"
            #         },
            #         {
            #             "name": "bus_tag",
            #             "value": "volvo",
            #             "key": "d0a60dc929de491a85a09fea59adb818"
            #         }
            #     ]
            # }
        """
        return self.clone(tags=self._tags.add_items(tags))

    def to_json(self, key_id_map: Optional[KeyIdMap] = None) -> Dict:
        """
        Convert the VideoObject to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: Json format as a dict
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_car = sly.ObjClass('vehicle', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)

            obj_car_json = video_obj_car.to_json()
            print(obj_car_json)
            # Output: {
            #     "key": "ce26e77a45bc45e88e3e17da1672d01f",
            #     "classTitle": "vehicle",
            #     "tags": []
            # }
        """
        data_json = {
            KEY: self.key().hex,
            LabelJsonFields.OBJ_CLASS_NAME: self.obj_class.name,
            LabelJsonFields.TAGS: self.tags.to_json(key_id_map),
        }

        if key_id_map is not None:
            item_id = key_id_map.get_object_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

        self._add_creation_info(data_json)
        return data_json

    @classmethod
    def from_json(cls, data: Dict, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None) -> VideoObject:
        """
        Convert a json dict to VideoObject. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Dict in json format.
        :type data: dict
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :raises: :class:`RuntimeError`, if object class name is not found in the given project meta
        :return: VideoObject object
        :rtype: :class:`VideoObject`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_car_json = {
                "classTitle": "vehicle",
                "tags": []
            }
            obj_class_car = sly.ObjClass('vehicle', sly.Rectangle)
            obj_collection = sly.ObjClassCollection([obj_class_car])
            meta = sly.ProjectMeta(obj_classes=obj_collection)

            video_obj_from_json = sly.VideoObject.from_json(obj_car_json, meta)
        """
        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        if obj_class is None:
            raise RuntimeError(
                f"Failed to deserialize a object from JSON: class name {obj_class_name!r} "
                f"was not found in the given project meta."
            )

        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_object(key, data.get(ID, None))

        class_id = data.get(CLASS_ID, None)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)

        return cls(
            obj_class=obj_class,
            key=key,
            tags=VideoTagCollection.from_json(
                data[LabelJsonFields.TAGS], project_meta.tag_metas
            ),
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def clone(self, obj_class: Optional[ObjClass]=None, tags: Optional[VideoTagCollection] = None, key: Optional[KeyIdMap]=None,
              class_id: Optional[int]=None, labeler_login: Optional[str]=None, updated_at: Optional[str]=None, created_at: Optional[str]=None) -> VideoObject:
        """
        Makes a copy of VideoObject with new fields, if fields are given, otherwise it will use fields of the original VideoObject.

        :param obj_class: VideoObject :class:`class<supervisely.annotation.obj_class.ObjClass>`.
        :type obj_class: ObjClass, optional
        :param tags: VideoObject :class:`tags<supervisely.video_annotation.video_tag_collection.VideoTagCollection>`.
        :type tags: VideoTagCollection, optional
        :param key: KeyIdMap object.
        :type key: KeyIdMap, optional
        :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which VideoObject belongs.
        :type class_id: int, optional
        :param labeler_login: Login of the user who created VideoObject.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when VideoObject was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when VideoObject was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :return: VideoObject object
        :rtype: :class:`VideoObject`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_car = sly.ObjClass('vehicle', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)

            # New ObjClass to clone
            obj_class_bus = sly.ObjClass('bus', sly.Rectangle)

            # New VideoTagCollection to clone
            from supervisely.video_annotation.video_tag import VideoTag
            meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
            car_tag = VideoTag(meta_car, value='acura')
            from supervisely.video_annotation.video_tag_collection import VideoTagCollection
            tags = VideoTagCollection([car_tag])
            # Clone
            new_obj_vehicle = video_obj_car.clone(obj_class=obj_class_bus, tags=tags)
            new_obj_vehicle_json = new_obj_vehicle.to_json()
            print(new_obj_vehicle_json)
            # Output: {
            #     "key": "39ae5b9ce1ca405c9f53544374b3f5be",
            #     "classTitle": "bus",
            #     "tags": [
            #         {
            #             "name": "car_tag",
            #             "value": "acura",
            #             "key": "3119b6e38fe24fe7a220e881154fd9ba"
            #         }
            #     ]
            # }
        """
        return self.__class__(obj_class=take_with_default(obj_class, self.obj_class),
                              key=take_with_default(key, self._key),
                              tags=take_with_default(tags, self.tags),
                              class_id=take_with_default(class_id, self.class_id),
                              labeler_login=take_with_default(labeler_login, self.labeler_login),
                              updated_at=take_with_default(updated_at, self.updated_at),
                              created_at=take_with_default(created_at, self.created_at))

