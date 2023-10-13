from __future__ import annotations
from typing import Optional, Dict, Union
import uuid

from supervisely._utils import take_with_default
from supervisely.annotation.tag import Tag, TagJsonFields
from supervisely.annotation.tag_meta import TagMeta
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.constants import KEY, ID


class PointcloudTag(Tag):
    """
    PointcloudTag object for :class:`PointcloudAnnotation<supervisely.pointcloud_annotation.pointcloud_annotation.PointcloudAnnotation>`. :class:`PointcloudTag<PointcloudTag>` object is immutable.

    :param meta: General information about Pointcloud Tag.
    :type meta: :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>`
    :param value: Pointcloud Tag value. Depends on :class:`TagValueType<supervisely.annotation.tag_meta.TagValueType>` of :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>`.
    :type value: :class:`str` or :class:`int` or :class:`float` or :class:`NoneType`, optional
    :param key: uuid.UUID object.
    :type key: uuid.UUID, optional
    :param sly_id: Video Tag ID in Supervisely.
    :type sly_id: :class:`int`, optional
    :param labeler_login: Login of user who created PointcloudTag.
    :type labeler_login: :class:`str`, optional
    :param updated_at: Date and Time when PointcloudTag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: :class:`str`, optional
    :param created_at: Date and Time when PointcloudTag was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: :class:`str`, optional
    :raises: :class:`ValueError`, If PointcloudTag value is incompatible to :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>` value type.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
        # Now we can create a PointcloudTag using our TagMeta
        tag_dog = sly.PointcloudTag(meta_dog)
        # When you are creating a new Tag
        # Tag.value is automatically cross-checked against your TagMeta value type to make sure the value is valid.
        # If we now try to add a value to our newly created Tag, we receive "ValueError", because our TagMeta value type is "NONE"
        tag_dog = sly.PointcloudTag(meta_dog, value="Husky")
        # Output: ValueError: Tag dog can not have value Husky

        # Let's create another Tag with a string value type
        meta_cat = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
        tag_cat = sly.PointcloudTag(meta_cat, value="Fluffy")

        # Now let's create a Tag using TagMeta with "ONEOF_STRING" value type
        # In order to use "oneof_string value type", you must initialize a variable with possible values(see class TagMeta for more information)
        colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
        meta_coat_color = sly.TagMeta('coat color', sly.TagValueType.ONEOF_STRING, possible_values=colors)
        tag_coat_color = sly.PointcloudTag(meta_coat_color, value="white")

        # If given value is not in a list of possible Tags, ValueError will be raised
        tag_coat_color = sly.PointcloudTag(meta_coat_color, value="yellow")
        # Output: ValueError: Tag coat color can not have value yellow
    """

    def __init__(
        self,
        meta: TagMeta,
        value: Optional[Union[str, int, float]] = None,
        key: Optional[uuid.UUID] = None,
        sly_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        super(PointcloudTag, self).__init__(
            meta,
            value=value,
            sly_id=sly_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

        self._key = take_with_default(key, uuid.uuid4())

    def get_compact_str(self) -> str:
        """
        Get string with information about PointcloudTag name and value.

        :return: Information about PointcloudTag object
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            meta_cat = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
            tag_cat = sly.PointcloudTag(meta_cat, value="Fluffy")
            compact_tag_cat = tag_cat.get_compact_str()

            print(compact_tag_cat) # cat:Fluffy
        """

        return super(PointcloudTag, self).get_compact_str()

    def __eq__(self, other: PointcloudTag) -> bool:
        """
        Checks that 2 Pointcloud Tags are equal by comparing their meta and value.

        :param other: PointcloudTag object.
        :type other: :class:`PointcloudTag<PointcloudTag>`
        :return: True if comparable objects are equal, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Let's create 2 identical Tags
            meta_lemon_1 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_1 = sly.PointcloudTag(meta_lemon_1)

            meta_lemon_2 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_2 = sly.PointcloudTag(meta_lemon_2)

            # and 1 different Tag to compare them
            meta_cucumber = sly.TagMeta('Cucumber', sly.TagValueType.ANY_STRING)
            tag_cucumber = sly.PointcloudTag(meta_cucumber, value="Fresh")

            # Compare identical Pointcloud Tags
            tag_lemon_1 == tag_lemon_2      # True

            # Compare unidentical Pointcloud Tags
            tag_lemon_1 == tag_cucumber     # False
        """

        return super(PointcloudTag, self).__eq__(other)

    def __ne__(self, other: PointcloudTag) -> bool:
        """
        Checks that 2 Pointcloud Tags are opposite.

        :param other: PointcloudTag object.
        :type other: :class:`PointcloudTag<PointcloudTag>`
        :return: True if comparable objects are not equal, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Let's create 2 identical Tags
            meta_lemon_1 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_1 = sly.PointcloudTag(meta_lemon_1)

            meta_lemon_2 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_2 = sly.PointcloudTag(meta_lemon_2)

            # and 1 different Tag to compare them
            meta_cucumber = sly.TagMeta('Cucumber', sly.TagValueType.ANY_STRING)
            tag_cucumber = sly.PointcloudTag(meta_cucumber, value="Fresh")

            # Compare identical Pointcloud Tags
            tag_lemon_1 != tag_lemon_2      # False

            # Compare unidentical Pointcloud Tags
            tag_lemon_1 != tag_cucumber     # True
        """

        return super(PointcloudTag, self).__ne__(other)

    def key(self) -> uuid.UUID:
        """
        Get PointcloudTag key value.

        :return: PointcloudTag key value
        :rtype: uuid.UUID
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            weather_conditions = ["Sunny", "Cloudy", "Snowy", "Foggy", "Rainy"]
            meta_weather = sly.TagMeta("weather", sly.TagValueType.ONEOF_STRING, possible_values=weather_conditions)
            tag_weather = sly.Tag(meta_weather, value="Sunny")
            key = tag_weather.key()

            print(key)
            # Output: 5c7988e0-eee4-4eb1-972c-b1e3e879f78c
        """

        return self._key

    def to_json(self, key_id_map: Optional[KeyIdMap] = None) -> Dict:
        """
        Convert the PointcloudTag to a json dict.
        Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            tag_dog = sly.PointcloudTag(meta_dog)
            tag_dog_json = tag_dog.to_json()

            print(tag_dog_json)
            # Output: {
            #     "name": "dog",
            #     "key": "058ad7993a534082b4d94cc52542a97d"
            # }
        """

        data_json = super(PointcloudTag, self).to_json()
        if type(data_json) is str:
            # case when tag has no value, super.to_json() returns tag name
            data_json = {TagJsonFields.TAG_NAME: data_json}
        data_json[KEY] = self.key().hex

        if key_id_map is not None:
            item_id = key_id_map.get_tag_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

        return data_json

    @classmethod
    def from_json(
        cls,
        data: Dict,
        tag_meta_collection: TagMetaCollection,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> PointcloudTag:
        """
        Convert a json dict to VideoTag. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: PointcloudTag in json format as a dict.
        :type data: :class:`dict`
        :param tag_meta_collection: TagMetaCollection object.
        :type tag_meta_collection: :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>`: TagMetaCollection
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: PointcloudTag object
        :rtype: :class:`PointcloudTag<PointcloudTag>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            tag_cat_json = {
                "name": "cat",
                "value": "Fluffy"
            }

            meta_cat = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
            meta_collection = sly.TagMetaCollection([meta_cat])
            tag_cat = sly.PointcloudTag.from_json(tag_cat_json, meta_collection)
        """

        temp = super(PointcloudTag, cls).from_json(data, tag_meta_collection)
        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_tag(key, data.get(ID, None))

        return cls(
            meta=temp.meta,
            value=temp.value,
            key=key,
            sly_id=temp.sly_id,
            labeler_login=temp.labeler_login,
            updated_at=temp.updated_at,
            created_at=temp.created_at,
        )

    def clone(
        self,
        meta: Optional[TagMeta] = None,
        value: Optional[Union[str, int, float]] = None,
        key: Optional[uuid.UUID] = None,
        sly_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> PointcloudTag:
        """
        Clone makes a copy of Pointcloud Tag with new fields, if fields are given, otherwise it will use original Tag fields.

        :param meta: TagMeta object.
        :type meta: :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>`, optional
        :param value: Pointcloud Tag value. Depends on :class:`TagValueType<supervisely.annotation.tag_meta.TagValueType>` of :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>`.
        :type value: :class:`str` or :class:`int` or :class:`float` or :class:`NoneType`, optional
        :param key: uuid.UUID object.
        :type key: uuid.UUID, optional
        :param sly_id: Pointcloud Tag ID in Supervisely server.
        :type sly_id: :class:`int`, optional
        :param labeler_login: Login of user who created Pointcloud Tag.
        :type labeler_login: :class:`str`, optional
        :param updated_at: Date and Time when Pointcloud Tag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: :class:`str`, optional
        :param created_at: Date and Time when Pointcloud Tag was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: :class:`str`, optional
        :return: New instance of Pointcloud Tag
        :rtype: :class:`PointcloudTag<PointcloudTag>`
        :raises: :class:`ValueError`, If PointcloudTag value is incompatible to :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>` value type.

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            # Original Pointcloud Tag
            weather_conditions = ["Sunny", "Cloudy", "Snowy", "Foggy", "Rainy"]
            meta_weather = sly.TagMeta("weather", sly.TagValueType.ONEOF_STRING, possible_values=weather_conditions)

            tag_weather = sly.PointcloudTag(meta_weather, value="Sunny")

            # Let's create some more tags by cloning our original Pointcloud Tag
            # Remember that PointcloudTag class object is immutable, and we need to assign new instance of PointcloudTag to a new variable
            clone_weather_1 = tag_weather.clone(value="Snowy")

            clone_weather_2 = tag_weather.clone(value="Cloudy")

            clone_weather_3 = tag_weather.clone(value="Rainy")
        """

        return PointcloudTag(
            meta=take_with_default(meta, self.meta),
            value=take_with_default(value, self.value),
            key=take_with_default(key, self.key()),
            sly_id=take_with_default(sly_id, self.sly_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )
