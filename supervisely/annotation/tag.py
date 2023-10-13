# coding: utf-8
"""Tag can be attached to whole image and/or to individual :class:`Label<supervisely.annotation.label.Label>`"""

# docs
from __future__ import annotations
from typing import List, Optional, Dict, Union
from supervisely.annotation.tag_meta import TagMeta
from supervisely.annotation.tag_meta_collection import TagMetaCollection


from supervisely.annotation.tag_meta import TagValueType
from supervisely.collection.key_indexed_collection import KeyObject
from supervisely._utils import take_with_default


class TagJsonFields:
    """Json fields for :class:`Annotation<supervisely.annotation.tag.Tag>`"""

    TAG_NAME = "name"
    """"""
    VALUE = "value"
    """"""
    LABELER_LOGIN = "labelerLogin"
    """"""
    UPDATED_AT = "updatedAt"
    """"""
    CREATED_AT = "createdAt"
    """"""
    ID = "id"
    """"""
    # TAG_META_ID = 'tagId'
    # """"""


class Tag(KeyObject):
    """
    :class:`Tag<Tag>` can be attached to whole image and/or to individual :class:`Label<LabelBase>`. :class:`Tag<Tag>` object is immutable.

    :param meta: General information about Tag.
    :type meta: TagMeta
    :param value: Tag value. Depends on :class:`TagValueType<TagValueType>` of :class:`TagMeta<TagMeta>`.
    :type value: Optional[Union[str, int, float]]
    :param sly_id: Tag ID in Supervisely server.
    :type sly_id: int, optional
    :param labeler_login: Login of user who created Tag.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Tag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Tag was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`, if meta is None or if Tag has incompatible value against it's meta value type
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Let's create 3 Tags with different values
        # First we need to initialize a TagMeta
        meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)

        # Now we can create a Tag using our TagMeta
        tag_dog = sly.Tag(meta_dog)

        # When you are creating a new Tag
        # Tag.value is automatically cross-checked against your TagMeta value type to make sure the value is valid.
        # If we now try to add a value to our newly created Tag, we receive "ValueError", because our TagMeta value type is "NONE"
        tag_dog = sly.Tag(meta_dog, value="Husky")
        # Output: ValueError: Tag dog can not have value Husky

        # Let's create another Tag with a string value type
        meta_cat = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
        tag_cat = sly.Tag(meta_cat, value="Fluffy")

        # Now let's create a Tag using TagMeta with "ONEOF_STRING" value type
        # In order to use "oneof_string value type", you must initialize a variable with possible values(see class TagMeta for more information)
        colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
        meta_coat_color = sly.TagMeta('coat color', sly.TagValueType.ONEOF_STRING, possible_values=colors)
        tag_coat_color = sly.Tag(meta_coat_color, value="white")

        # If given value is not in a list of possible Tags, ValueError will be raised
        tag_coat_color = sly.Tag(meta_coat_color, value="yellow")
        # Output: ValueError: Tag coat color can not have value yellow
    """

    def __init__(
        self,
        meta: TagMeta,
        value: Optional[Union[str, int, float]] = None,
        sly_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        if meta is None:
            raise ValueError("TagMeta is None")
        self._meta = meta
        self._value = value
        if not self._meta.is_valid_value(value):
            raise ValueError("Tag {} can not have value {}".format(self.meta.name, value))
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at
        self.sly_id = sly_id

    @property
    def meta(self) -> TagMeta:
        """
        General information about Tag. When creating a new Tag, it's value is automatically cross-checked against :class:`TagValueType<supervisely.annotation.tag_meta.TagValueType>` to make sure that value is valid.

        :return: TagMeta object
        :rtype: :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            tag_dog = sly.Tag(meta_dog)

            # Our TagMeta has value type 'NONE', if we try to add value to our Tag, "ValueError" error will be raised
            tag_dog = sly.Tag(meta_dog, value="Husky")
            # Output: ValueError: Tag dog can not have value Husky
        """
        return self._meta

    @property
    def value(self) -> str or int or float:
        """
        Tag value. Return type depends on :class:`TagValueType<supervisely.annotation.tag_meta.TagValueType>`.

        :return: Tag value
        :rtype: :class:`str`, :class:`int` or :class:`float` or :class:`None`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            tag_dog = sly.Tag(meta_dog, value="Husky")

            meta_age = sly.TagMeta('age', sly.TagValueType.ANY_NUMBER)
            tag_age = sly.Tag(meta_age, value=9)

            colors = ["Black", "White", "Golden", "Brown"]
            meta_color = sly.TagMeta('coat color', sly.TagValueType.ONEOF_STRING, possible_values=colors)
            tag_color = sly.Tag(meta_color, value="White")

            type(tag_dog.value)   # 'str'
            type(tag_age.value)   # 'int'
            type(tag_color.value) # 'str'
        """
        return self._value

    @property
    def name(self) -> str:
        """
        Name property.

        :return: Name
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            tag_dog = sly.Tag(meta_dog, value="Husky")

            print(tag_dog.name)
            # Output: "dog"
        """
        return self._meta.name

    def key(self):
        """
        Get TagMeta key value.

        :return: TagMeta key value
        :rtype: str
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            weather_conditions = ["Sunny", "Cloudy", "Snowy", "Foggy", "Rainy"]
            meta_weather = sly.TagMeta("weather", sly.TagValueType.ONEOF_STRING, possible_values=weather_conditions)
            tag_weather = sly.Tag(meta_weather, value="Sunny")
            key = tag_weather.key()
            print(key)
            # Output: weather
        """

        return self._meta.key()

    def to_json(self) -> Dict:
        """
        Convert the Tag to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Tag with all fields filled in
            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            tag_dog = sly.Tag(meta=meta_dog, value="Husky", sly_id=38456, labeler_login="admin",
                              updated_at="2021-01-22T19:37:50.158Z", created_at="2021-01-22T18:00:00.000Z")

            tag_dog_json = tag_dog.to_json()
            print(tag_dog_json)
            # Notice that sly_id won't print
            # Output: {
            #  "name": "dog",
            #  "value": "Husky",
            #  "labelerLogin": "admin",
            #  "updatedAt": "2021-01-22T19:37:50.158Z",
            #  "createdAt": "2021-01-22T18:00:00.000Z"
            # }
        """
        res = {
            TagJsonFields.TAG_NAME: self.meta.name
        }
        if self.meta.value_type != TagValueType.NONE:
            res[TagJsonFields.VALUE] = self.value
        if self.labeler_login is not None:
            res[TagJsonFields.LABELER_LOGIN] = self.labeler_login
        if self.updated_at is not None:
            res[TagJsonFields.UPDATED_AT] = self.updated_at
        if self.created_at is not None:
            res[TagJsonFields.CREATED_AT] = self.created_at
        return res

    @classmethod
    def from_json(cls, data: Dict, tag_meta_collection: TagMetaCollection) -> Tag:
        """
        Convert a json dict to Tag. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Tag in json format as a dict.
        :type data: dict
        :param tag_meta_collection: TagMetaCollection object.
        :type tag_meta_collection: TagMetaCollection
        :return: Tag object
        :rtype: :class:`Tag<Tag>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            tag_metas = sly.TagMetaCollection([meta_dog])

            data = {
                "name": "dog",
                "value": "Husky",
                "labelerLogin": "admin",
                "updatedAt": "2021-01-22T19:37:50.158Z",
                "createdAt": "2021-01-22T18:00:00.000Z"
            }

            tag_dog = sly.Tag.from_json(data, tag_metas)
        """
        if type(data) is str:
            tag_name = data
            value = None
            labeler_login = None
            updated_at = None
            created_at = None
            sly_id = None
        else:
            tag_name = data[TagJsonFields.TAG_NAME]
            value = data.get(TagJsonFields.VALUE, None)
            labeler_login = data.get(TagJsonFields.LABELER_LOGIN, None)
            updated_at = data.get(TagJsonFields.UPDATED_AT, None)
            created_at = data.get(TagJsonFields.CREATED_AT, None)
            sly_id = data.get(TagJsonFields.ID, None)
        meta = tag_meta_collection.get(tag_name)
        return cls(
            meta=meta,
            value=value,
            sly_id=sly_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def get_compact_str(self) -> str:
        """
        Displays information about Tag's name and value in string format.

        :return: Name and value of the given Tag
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            tag_dog = sly.Tag(meta_dog, value="Husky")

            print(tag_dog.get_compact_str())
            # Output: 'dog:Husky'
        """
        if (self.meta.value_type != TagValueType.NONE) and (len(str(self.value)) > 0):
            return "{}:{}".format(self.name, self.value)
        return self.name

    def __eq__(self, other: Tag) -> bool:
        """
        Checks that 2 Tags are equal by comparing their meta and value.

        :param other: Tag object.
        :type other: Tag
        :return: True if comparable objects are equal, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Let's create 2 identical Tags
            meta_lemon_1 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_1 = sly.Tag(meta_lemon_1)

            meta_lemon_2 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_2 = sly.Tag(meta_lemon_2)

            # and 1 different Tag to compare them
            meta_cucumber = sly.TagMeta('Cucumber', sly.TagValueType.ANY_STRING)
            tag_cucumber = sly.Tag(meta_cucumber, value="Fresh")

            # Compare identical Tags
            tag_lemon_1 == tag_lemon_2      # True

            # Compare unidentical Tags
            tag_lemon_1 == tag_cucumber     # False
        """
        return isinstance(other, Tag) and self.meta == other.meta and self.value == other.value

    def __ne__(self, other: Tag) -> bool:
        """
        Checks that 2 Tags are opposite.

        :param other: Tag object.
        :type other: Tag
        :return: True if comparable objects are not equal, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Let's create 2 identical Tags
            meta_lemon_1 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_1 = sly.Tag(meta_lemon_1)

            meta_lemon_2 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            tag_lemon_2 = sly.Tag(meta_lemon_2)

            # and 1 different Tag to compare them
            meta_cucumber = sly.TagMeta('Cucumber', sly.TagValueType.ANY_STRING)
            tag_cucumber = sly.Tag(meta_cucumber, value="Fresh")

            # Compare identical Tags
            tag_lemon_1 != tag_lemon_2      # False

            # Compare unidentical Tags
            tag_lemon_1 != tag_cucumber     # True
        """
        return not self == other

    def clone(
        self,
        meta: Optional[TagMeta] = None,
        value: Optional[Union[str, int, float]] = None,
        sly_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Tag:
        """
        Clone makes a copy of Tag with new fields, if fields are given, otherwise it will use original Tag fields.

        :param meta: General information about Tag.
        :type meta: TagMeta
        :type value: str or int or float or None
        :param sly_id: Tag ID in Supervisely server.
        :type sly_id: int, optional
        :param labeler_login: Login of user who created Tag.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when Tag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when Tag was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :return: New instance of Tag
        :rtype: :class:`Tag<Tag>`
        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            #Original Tag
            weather_conditions = ["Sunny", "Cloudy", "Snowy", "Foggy", "Rainy"]
            meta_weather = sly.TagMeta("weather", sly.TagValueType.ONEOF_STRING, possible_values=weather_conditions)

            tag_weather = sly.Tag(meta_weather, value="Sunny")

            # Let's create some more tags by cloning our original Tag
            # Remember that Tag class object is immutable, and we need to assign new instance of Tag to a new variable
            clone_weather_1 = tag_weather.clone(value="Snowy")

            clone_weather_2 = tag_weather.clone(value="Cloudy")

            clone_weather_3 = tag_weather.clone(value="Rainy")
        """
        return Tag(
            meta=take_with_default(meta, self.meta),
            value=take_with_default(value, self.value),
            sly_id=take_with_default(sly_id, self.sly_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )

    def __str__(self):
        return "{:<7s}{:<10}{:<7s} {:<13}{:<7s} {:<10}".format(
            "Name:",
            self._meta.name,
            "Value type:",
            self._meta.value_type,
            "Value:",
            str(self.value),
        )

    @classmethod
    def get_header_ptable(cls):
        """
        Get header of the table with tags.

        :return: List of table header values.
        :rtype: List[str]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            header = sly.Tag.get_header_ptable()

            print(header)
            # Output: ['Name', 'Value type', 'Value']
        """

        return ["Name", "Value type", "Value"]

    def get_row_ptable(self):
        """
        Get row with tag properties.

        :return: List of tag properties.
        :rtype: List[str]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            weather_conditions = ["Sunny", "Cloudy", "Snowy", "Foggy", "Rainy"]
            meta_weather = sly.TagMeta("weather", sly.TagValueType.ONEOF_STRING, possible_values=weather_conditions)
            tag_weather = sly.Tag(meta_weather, value="Sunny")

            row = tag_weather.get_row_ptable()

            print(row)
            # Output: ['weather', 'oneof_string', 'Sunny']
        """

        return [self._meta.name, self._meta.value_type, self.value]

    def _set_id(self, id: int):
        self.sly_id = id

    def _set_updated_at(self, updated_at: str):
        self.updated_at = updated_at

    def _set_created_at(self, created_at: str):
        self.created_at = created_at

    def _set_labeler_login(self, labeler_login: str):
        self.labeler_login = labeler_login
