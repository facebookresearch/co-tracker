# coding: utf-8

# docs
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Union
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.annotation.tag_meta import TagMeta


import uuid
from supervisely.annotation.tag import Tag, TagJsonFields
from supervisely._utils import take_with_default
from supervisely.video_annotation.constants import KEY, ID, FRAME_RANGE
from supervisely.video_annotation.key_id_map import KeyIdMap


class VideoTag(Tag):
    """
    VideoTag object for :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`. :class:`VideoTag<VideoTag>` object is immutable.

    :param meta: General information about Video Tag.
    :type meta: TagMeta
    :param value: Video Tag value. Depends on :class:`TagValueType<TagValueType>` of :class:`TagMeta<TagMeta>`.
    :type value: str or int or float or None, optional
    :param frame_range: Video Tag frame range.
    :type frame_range: Tuple[int, int] or List[int, int], optional
    :param key: uuid.UUID object.
    :type key: uuid.UUID, optional
    :param sly_id: Video Tag ID in Supervisely.
    :type sly_id: int, optional
    :param labeler_login: Login of user who created VideoTag.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when VideoTag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when VideoTag was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.video_annotation.video_tag import VideoTag

        meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
        # Now we can create a VideoTag using our TagMeta
        tag_dog = VideoTag(meta_dog)
        # When you are creating a new Tag
        # Tag.value is automatically cross-checked against your TagMeta value type to make sure the value is valid.
        # If we now try to add a value to our newly created Tag, we receive "ValueError", because our TagMeta value type is "NONE"
        tag_dog = VideoTag(meta_dog, value="Husky")
        # Output: ValueError: Tag dog can not have value Husky

        # Let's create another Tag with a string value type and frame range
        meta_cat = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
        tag_cat = VideoTag(meta_cat, value="Fluffy", frame_range=(5, 10))

        # Now let's create a Tag using TagMeta with "ONEOF_STRING" value type
        # In order to use "oneof_string value type", you must initialize a variable with possible values(see class TagMeta for more information)
        colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
        meta_coat_color = sly.TagMeta('coat color', sly.TagValueType.ONEOF_STRING, possible_values=colors)
        tag_coat_color = VideoTag(meta_coat_color, value="white", frame_range=(15, 20))

        # If given value is not in a list of possible Tags, ValueError will be raised
        tag_coat_color = VideoTag(meta_coat_color, value="yellow")
        # Output: ValueError: Tag coat color can not have value yellow
    """
    def __init__(self, meta: TagMeta, value: Optional[Union[str, int, float]]=None, frame_range: Optional[Tuple[int, int]]=None,
                 key: Optional[uuid.UUID]=None, sly_id: Optional[int]=None, labeler_login: Optional[str]=None,
                 updated_at: Optional[str]=None, created_at: Optional[str]=None):
        super(VideoTag, self).__init__(meta, value=value, sly_id=sly_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
        
        self._frame_range = None
        if frame_range is not None:
            if not isinstance(frame_range, (tuple, list)):
                raise TypeError('frame_range has to be a tuple or a list. Given type "{}".'.format(type(frame_range)))

            if len(frame_range) != 2 or not isinstance(frame_range[0], int) or not isinstance(frame_range[1], int):
                raise ValueError("frame_range has to be a tuple or a list with 2 int values.") 
            self._frame_range = list(frame_range)

        self._key = take_with_default(key, uuid.uuid4())

    @property
    def frame_range(self) -> Tuple[int, int]:
        """
        VideoTag frame range.

        :return: Range of frames for current VideoTag
        :rtype: :class:`Tuple[int, int]`
        :Usage example:

         .. code-block:: python

            cat_range = tag_cat.frame_range # [5, 10]
        """
        return self._frame_range

    def key(self) -> uuid.UUID:
        return self._key

    def to_json(self, key_id_map: Optional[KeyIdMap] = None) -> Dict:
        """
        Convert the VideoTag to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: Key ID Map object.
        :type key_id_map: KeyIdMap, optional
        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.video_tag import VideoTag
            meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            tag_dog = VideoTag(meta_dog)
            tag_dog_json = tag_dog.to_json()
            print(tag_dog_json)
            # Output: {
            #     "name": "dog",
            #     "key": "058ad7993a534082b4d94cc52542a97d"
            # }
        """
        data_json = super(VideoTag, self).to_json()
        if type(data_json) is str:
            # @TODO: case when tag has no value, super.to_json() returns tag name
            data_json = {TagJsonFields.TAG_NAME: data_json}
        if self.frame_range is not None:
            data_json[FRAME_RANGE] = self.frame_range
        data_json[KEY] = self.key().hex

        if key_id_map is not None:
            item_id = key_id_map.get_tag_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

        return data_json

    @classmethod
    def from_json(cls, data: Dict, tag_meta_collection: TagMetaCollection, key_id_map: Optional[KeyIdMap] = None) -> VideoTag:
        """
        Convert a json dict to VideoTag. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: VideoTag in json format as a dict.
        :type data: dict
        :param tag_meta_collection: :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` object.
        :type tag_meta_collection: TagMetaCollection
        :param key_id_map: Key ID Map object.
        :type key_id_map: KeyIdMap, optional
        :return: VideoTag object
        :rtype: :class:`VideoTag<VideoTag>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            tag_cat_json = {
                "name": "cat",
                "value": "Fluffy",
                "frameRange": [
                    5,
                    10
                ]
            }

            from supervisely.video_annotation.video_tag import VideoTag
            meta_cat = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
            meta_collection = sly.TagMetaCollection([meta_cat])
            tag_cat = VideoTag.from_json(tag_cat_json, meta_collection)
        """
        temp = super(VideoTag, cls).from_json(data, tag_meta_collection)
        frame_range = data.get(FRAME_RANGE, None)
        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_tag(key, data.get(ID, None))

        return cls(meta=temp.meta, value=temp.value, frame_range=frame_range, key=key,
                   sly_id=temp.sly_id, labeler_login=temp.labeler_login, updated_at=temp.updated_at, created_at=temp.created_at)

    def get_compact_str(self) -> str:
        """
        Get string with information about VideoTag: name, value and range of frames.

        :return: Information about VideoTag object
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.video_tag import VideoTag
            meta_cat = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
            tag_cat = VideoTag(meta_cat, value="Fluffy", frame_range=(5, 10))
            compact_tag_cat = tag_cat.get_compact_str()
            print(compact_tag_cat) # cat:Fluffy[5 - 10]
        """
        res = super(VideoTag, self).get_compact_str()
        if self.frame_range is not None:
            res = "{}[{} - {}]".format(res, self.frame_range[0], self.frame_range[1])
        return res

    def __eq__(self, other: VideoTag) -> bool:
        """
        Checks that 2 VideoTags are equal by comparing their meta, value and frame_range.

        :param other: VideoTag object.
        :type other: VideoTag
        :return: True if comparable objects are equal, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.video_tag import VideoTag

            # Let's create 2 identical Tags
            meta_lemon_1 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_1 = VideoTag(meta_lemon_1)

            meta_lemon_2 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_2 = VideoTag(meta_lemon_2)

            # and 1 different Tag to compare them
            meta_cucumber = sly.TagMeta('Cucumber', sly.TagValueType.ANY_STRING)
            tag_cucumber = VideoTag(meta_cucumber, value="Fresh")

            # Compare identical Tags
            print(tag_lemon_1 == tag_lemon_2)      # True

            # Compare unidentical Tags
            print(tag_lemon_1 == tag_cucumber)     # False
        """
        return isinstance(other, VideoTag) and \
               self.meta == other.meta and \
               self.value == other.value and \
               self.frame_range == other.frame_range

    def clone(self, meta: Optional[TagMeta] = None, value: Optional[Union[str, int, float]] = None, frame_range: Optional[Tuple[int, int]] = None,
              key: Optional[uuid.UUID] = None, sly_id: Optional[int] = None, labeler_login: Optional[str] = None,
              updated_at: Optional[str] = None, created_at: Optional[str] = None) -> VideoTag:
        """
        Makes a copy of VideoTag with new fields, if fields are given, otherwise it will use fields of the original VideoTag.

        :param meta: General information about VideoTag.
        :type meta: TagMeta, optional
        :param value: VideoTag value. Depends on :class:`TagValueType<TagValueType>` of :class:`TagMeta<TagMeta>`.
        :type value: str or int or float or None, optional
        :param frame_range: VideoTag frame range.
        :type frame_range: Tuple[int, int] or List[int, int], optional
        :param key: uuid.UUID object.
        :type key: uuid.UUID, optional
        :param sly_id: VideoTag ID in Supervisely.
        :type sly_id: int, optional
        :param labeler_login: Login of user who created VideoTag.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when VideoTag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when VideoTag was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.video_tag import VideoTag

            meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
            car_tag = VideoTag(meta_car, value='acura', frame_range=(7, 9))

            meta_bus = sly.TagMeta('bus', sly.TagValueType.ANY_STRING)
            new_tag = car_tag.clone(meta=meta_bus, frame_range=(15, 129), key=car_tag.key())
            new_tag_json = new_tag.to_json()
            print(new_tag_json)
            # Output: {
            #     "name": "bus",
            #     "value": "acura",
            #     "frameRange": [
            #         15,
            #         129
            #     ],
            #     "key": "360438485fd34264921ca19bd43b0b71"
            # }
        """
        return self.__class__(
            meta=take_with_default(meta, self.meta),
            value=take_with_default(value, self.value),
            frame_range=take_with_default(frame_range, self.frame_range),
            key=take_with_default(key, self.key()),
            sly_id=take_with_default(sly_id, self.sly_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at)
        )

    def __str__(self):
        return '{:<7s}{:<10}{:<7s} {:<13}{:<7s} {:<10} {:<12}'.format('Name:', self._meta.name,
                                                               'Value type:', self._meta.value_type,
                                                               'Value:', str(self.value),
                                                               'FrameRange', str(self.frame_range))

    @classmethod
    def get_header_ptable(cls) -> List[str]:
        return ['Name', 'Value type', 'Value', 'Frame range']

    def get_row_ptable(self) -> List[str]:
        return [self._meta.name, self._meta.value_type, self.value, self.frame_range]
