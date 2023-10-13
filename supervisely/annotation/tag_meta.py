# coding: utf-8
"""General information about :class:`Tag<supervisely.annotation.tag.Tag>`"""

from __future__ import annotations
from typing import List, Optional, Dict
from copy import deepcopy
from supervisely.imaging.color import random_rgb, rgb2hex, hex2rgb, _validate_color
from supervisely.io.json import JsonSerializable
from supervisely.collection.key_indexed_collection import KeyObject
from supervisely._utils import take_with_default


class TagValueType:
    """
    Restricts Tag to have a certain value type.
    """

    NONE = "none"
    """"""
    ANY_NUMBER = "any_number"
    """"""
    ANY_STRING = "any_string"
    """"""
    ONEOF_STRING = "oneof_string"
    """"""


class TagMetaJsonFields:
    """
    Json fields for :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>`
    """

    ID = "id"
    """"""

    NAME = "name"
    """"""
    VALUE_TYPE = "value_type"
    """"""
    VALUES = "values"
    """"""
    COLOR = "color"
    """"""
    APPLICABLE_TYPE = "applicable_type"
    """"""
    HOTKEY = "hotkey"
    """"""
    APPLICABLE_CLASSES = "classes"
    """"""


class TagApplicableTo:
    """
    Defines Tag applicability only to images, objects or both.
    """

    ALL = "all"  # both images and objects
    """"""
    IMAGES_ONLY = "imagesOnly"
    """"""
    OBJECTS_ONLY = "objectsOnly"
    """"""


SUPPORTED_TAG_VALUE_TYPES = [
    TagValueType.NONE,
    TagValueType.ANY_NUMBER,
    TagValueType.ANY_STRING,
    TagValueType.ONEOF_STRING,
]
SUPPORTED_APPLICABLE_TO = [
    TagApplicableTo.ALL,
    TagApplicableTo.IMAGES_ONLY,
    TagApplicableTo.OBJECTS_ONLY,
]


class TagMeta(KeyObject, JsonSerializable):
    """
    General information about :class:`Tag<supervisely.annotation.tag>`. :class:`TagMeta<TagMeta>` object is immutable.

    :param name: Tag name.
    :type name: str
    :param value_type: Tag value type.
    :type value_type: str
    :param possible_values: List of possible values.
    :type possible_values: List[str], optional
    :param color: :class:`[R, G, B]` color, generates random color by default.
    :type color: List[int, int, int], optional
    :param sly_id: Tag ID in Supervisely server.
    :type sly_id: int, optional
    :param hotkey: Hotkey for Tag in annotation tool UI.
    :type hotkey: str, optional
    :param applicable_to: Defines applicability of Tag only to images, objects or both.
    :type applicable_to: str, optional
    :param applicable_classes: Defines applicability of Tag only to certain classes.
    :type applicable_classes: List[str], optional
    :raises: :class:`ValueError`, if color is not list, or doesn't have exactly 3 values
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # TagMeta
        meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)

        # TagMeta applicable only to Images example
        meta_cat = sly.TagMeta('cat', sly.TagValueType.NONE, applicable_to=sly.TagApplicableTo.IMAGES_ONLY)

        # TagMeta with string value applicable only to Objects example
        meta_breed = sly.TagMeta('breed', sly.TagValueType.ANY_STRING, applicable_to=sly.TagApplicableTo.OBJECTS_ONLY)

        # More complex TagMeta example
        # Create a list with possible values in order to use "ONEOF_STRING" value type
        coat_colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
        # Note that "ONEOF_STRING" value type requires possible values, otherwise ValueError will be raised
        meta_coat_color = sly.TagMeta('coat color', sly.TagValueType.ONEOF_STRING, coat_colors, [255,120,0], hotkey="M", applicable_to=sly.TagApplicableTo.OBJECTS_ONLY, applicable_classes=["dog", "cat"])
    """

    def __init__(
        self,
        name: str,
        value_type: str,
        possible_values: Optional[List[str]] = None,
        color: Optional[List[int]] = None,
        sly_id: Optional[int] = None,
        hotkey: Optional[str] = None,
        applicable_to: Optional[str] = None,
        applicable_classes: Optional[List[str]] = None,
    ):
        if value_type not in SUPPORTED_TAG_VALUE_TYPES:
            raise ValueError(
                "value_type = {!r} is unknown, should be one of {}".format(
                    value_type, SUPPORTED_TAG_VALUE_TYPES
                )
            )

        self._name = name
        self._value_type = value_type
        self._possible_values = possible_values
        self._color = random_rgb() if color is None else deepcopy(color)
        self._sly_id = sly_id
        self._hotkey = take_with_default(hotkey, "")
        self._applicable_to = take_with_default(applicable_to, TagApplicableTo.ALL)
        self._applicable_classes = take_with_default(applicable_classes, [])
        if self._applicable_to not in SUPPORTED_APPLICABLE_TO:
            raise ValueError(
                "applicable_to = {!r} is unknown, should be one of {}".format(
                    self._applicable_to, SUPPORTED_APPLICABLE_TO
                )
            )

        if self._value_type == TagValueType.ONEOF_STRING:
            if self._possible_values is None:
                raise ValueError(
                    "TagValueType is ONEOF_STRING. List of possible values have to be defined."
                )
            if not all(isinstance(item, str) for item in self._possible_values):
                raise ValueError(
                    "TagValueType is ONEOF_STRING. All possible values have to be strings"
                )
        elif self._possible_values is not None:
            raise ValueError(
                "TagValueType is {!r}. possible_values variable have to be None".format(
                    self._value_type
                )
            )

        _validate_color(self._color)

    @property
    def name(self) -> str:
        """
        Name.

        :return: Name
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            print(meta_dog.name)
            # Output: 'dog'
        """
        return self._name

    def key(self) -> str:
        return self.name

    @property
    def value_type(self) -> str:
        """
        Value type. See possible value types in :class:`TagValueType<TagValueType>`.

        :return: Value type
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            meta_dog.value_type == sly.TagValueType.ANY_STRING # True

            print(meta_dog.value_type)
            # Output: 'any_string'
        """
        return self._value_type

    @property
    def possible_values(self) -> List[str]:
        """
        Possible values of object. This is a required field if object has "oneof_string" value type.

        :raise: :class:`ValueError` if list of possible values is not defined or TagMeta value_type is not "oneof_string".
        :return: List of possible values
        :rtype: :class:`List[str]`
        :Usage example:

         .. code-block:: python

            # List of possible values
            coat_colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]

            # TagMeta
            meta_coat_color = sly.TagMeta('coat color', sly.TagValueType.ONEOF_STRING, possible_values=coat_colors)

            print(meta_coat_color.possible_values)
            # Output: ['brown', 'white', 'black', 'red', 'chocolate', 'gold', 'grey']

            # Note that this is a required field if object has "oneof_string" value type.
            meta_coat_color = sly.TagMeta('coat color', sly.TagValueType.ONEOF_STRING)
            # Output: ValueError: TagValueType is ONEOF_STRING. List of possible values have to be defined.
        """
        return self._possible_values.copy() if self._possible_values is not None else None

    @property
    def color(self) -> List[int, int, int]:
        """
        :class:`[R,G,B]` color.

        :return: Color
        :rtype: :class:`List[int, int, int]`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE, color=[255,120,0])

            print(meta_dog.color)
            # Output: [255,120,0]
        """
        return self._color.copy()

    @property
    def sly_id(self) -> int:
        """
        Tag ID in Supervisely server.

        :return: ID
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE, sly_id=38584)

            print(meta_dog.sly_id)
            # Output: 38584
        """
        return self._sly_id

    @property
    def hotkey(self) -> str:
        """
        Hotkey for Tag in annotation tool UI.

        :return: Hotkey
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE, hotkey='M')

            print(meta_dog.hotkey)
            # Output: 'M'
        """
        return self._hotkey

    @property
    def applicable_to(self) -> str:
        """
        Tag applicability to objects, images, or both.

        :return: Applicability
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE, applicable_to=IMAGES_ONLY)

            print(meta_dog.applicable_to)
            # Output: 'imagesOnly'
        """
        return self._applicable_to

    @property
    def applicable_classes(self) -> List[str]:
        """
        Applicable classes.

        :returns: List of applicable classes
        :rtype: :class:`List[str]`
        :Usage example:

         .. code-block:: python

            # Imagine we have 2 ObjClasses in our Project
            class_car = sly.ObjClass(name='car', geometry_type='rectangle')
            class_bicycle = sly.ObjClass(name='bicycle', geometry_type='rectangle')

            # You can put a "string" with ObjClass name or use ObjClass.name
            meta_vehicle = sly.TagMeta('vehicle', sly.TagValueType.NONE, applicable_classes=["car", class_bicycle.name])

            print(meta_vehicle.applicable_classes)
            # Output: ['car', 'bicycle']
        """
        return self._applicable_classes

    def to_json(self) -> Dict:
        """
        Convert the TagMeta to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            colors = ["brown", "white", "black", "red", "blue", "yellow", "grey"]
            meta_color = sly.TagMeta('Color',
                                    sly.TagValueType.ONEOF_STRING,
                                    possible_values=colors,
                                    color=[255, 120, 0],
                                    hotkey="M",
                                    applicable_classes=["car", "bicycle"])


            meta_color_json = meta_color.to_json()
            print(meta_color_json)
            # Output: {
            #     "name":"Color",
            #     "value_type":"oneof_string",
            #     "color":"#FF7800",
            #     "values":[
            #         "brown",
            #         "white",
            #         "black",
            #         "red",
            #         "blue",
            #         "yellow",
            #         "grey"
            #     ],
            #     "hotkey":"M",
            #     "applicable_type":"all",
            #     "classes":[
            #         "car",
            #         "bicycle"
            #     ]
            # }
        """
        jdict = {
            TagMetaJsonFields.NAME: self.name,
            TagMetaJsonFields.VALUE_TYPE: self.value_type,
            TagMetaJsonFields.COLOR: rgb2hex(self.color),
        }
        if self.value_type == TagValueType.ONEOF_STRING:
            jdict[TagMetaJsonFields.VALUES] = self.possible_values

        if self.sly_id is not None:
            jdict[TagMetaJsonFields.ID] = self.sly_id
        if self._hotkey is not None:
            jdict[TagMetaJsonFields.HOTKEY] = self.hotkey
        if self._applicable_to is not None:
            jdict[TagMetaJsonFields.APPLICABLE_TYPE] = self.applicable_to
        if self._applicable_classes is not None:
            jdict[TagMetaJsonFields.APPLICABLE_CLASSES] = self.applicable_classes

        return jdict

    @classmethod
    def from_json(cls, data: Dict) -> TagMeta:
        """
        Convert a json dict to TagMeta. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: TagMeta in json format as a dict.
        :type data: dict
        :return: TagMeta object
        :rtype: :class:`TagMeta<TagMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            data = {
                "name":"Color",
                "value_type":"oneof_string",
                "color":"#FF7800",
                "values":[
                    "brown",
                    "white",
                    "black",
                    "red",
                    "blue",
                    "yellow",
                    "grey"
                ],
                "hotkey":"M",
                "applicable_type":"all",
                "classes":[
                    "car",
                    "bicycle"
                ]
            }

            meta_colors = sly.TagMeta.from_json(data)
        """
        if isinstance(data, str):
            return cls(name=data, value_type=TagValueType.NONE)
        elif isinstance(data, dict):
            name = data[TagMetaJsonFields.NAME]
            value_type = data[TagMetaJsonFields.VALUE_TYPE]
            values = data.get(TagMetaJsonFields.VALUES)
            color = data.get(TagMetaJsonFields.COLOR)
            if color is not None:
                color = hex2rgb(color)
            sly_id = data.get(TagMetaJsonFields.ID, None)

            hotkey = data.get(TagMetaJsonFields.HOTKEY, "")
            applicable_to = data.get(TagMetaJsonFields.APPLICABLE_TYPE, TagApplicableTo.ALL)
            applicable_classes = data.get(TagMetaJsonFields.APPLICABLE_CLASSES, [])

            return cls(
                name=name,
                value_type=value_type,
                possible_values=values,
                color=color,
                sly_id=sly_id,
                hotkey=hotkey,
                applicable_to=applicable_to,
                applicable_classes=applicable_classes,
            )
        else:
            raise ValueError("Tags must be dict or str types.")

    def add_possible_value(self, value: str) -> TagMeta:
        """
        Adds a new value to the list of possible values.

        :param value: New value that will be added to a list.
        :type value: str
        :raises: :class:`ValueError`, if object's value type is not "oneof_string" or already exists in a list
        :return: New instance of TagMeta
        :rtype: :class:`TagMeta<TagMeta>`
        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            #In order to add possible values, you must first initialize a variable where all possible values will be stored if it doesnt exist already
            colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
            meta_coat_color = sly.TagMeta('coat color', sly.TagValueType.ONEOF_STRING, possible_values=colors, applicable_classes=["dog", "cat"])

            print(meta_coat_color.possible_values)
            # Output: ['brown', 'white', 'black', 'red', 'chocolate', 'gold', 'grey']

            #Now we can add new possible value to our TagMeta
            # Remember that TagMeta object is immutable, and we need to assign new instance of TagMeta to a new variable
            meta_coat_color = meta_coat_color.add_possible_value("bald (no coat)")

            print(meta_coat_color.possible_values)
            # Output: ['brown', 'white', 'black', 'red', 'chocolate', 'gold', 'grey', 'bald (no coat)']
        """
        if self.value_type is TagValueType.ONEOF_STRING:
            if value in self._possible_values:
                raise ValueError("Value {} already exists for tag {}".format(value, self.name))
            else:
                return self.clone(possible_values=[*self.possible_values, value])
        else:
            raise ValueError(
                "Tag {!r} has type {!r}. Possible value can be added only to oneof_string".format(
                    self.name, self.value_type
                )
            )

    def is_valid_value(self, value: str) -> bool:
        """
        Checks value against object value type to make sure that value is valid.

        :param value: Value to check.
        :type value: str
        :return: True if value is supported, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Initialize TagMeta
            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)

            # Check what value type is in our Tagmeta
            print(meta_dog.value_type)
            # Output: 'any_string'

            # Our TagMeta has 'any_string' value type, it means only 'string' values will work with it
            # Let's check if value is valid for our TagMeta
            meta_dog.is_valid_value('Woof!')            # True
            meta_dog.is_valid_value(555)                # False

            # TagMetas with 'any_number' value type are compatible with 'int' and 'float' values
            meta_quantity = sly.TagMeta('quantity', sly.TagValueType.ANY_NUMBER)

            meta_quantity.is_valid_value('new string value') # False
            meta_quantity.is_valid_value(555)                # True
            meta_quantity.is_valid_value(3.14159265359)      # True
        """
        if self.value_type == TagValueType.NONE:
            return value is None
        elif self.value_type == TagValueType.ANY_NUMBER:
            return isinstance(value, (int, float))
        elif self.value_type == TagValueType.ANY_STRING:
            return isinstance(value, str)
        elif self.value_type == TagValueType.ONEOF_STRING:
            return isinstance(value, str) and (value in self._possible_values)
        else:
            raise ValueError("Unsupported TagValueType detected ({})!".format(self.value_type))

    def __eq__(self, other: TagMeta) -> bool:
        """
        Checks that 2 TagMetas are equal by their name, value type and possible values.

        :param other: TagMeta object.
        :type other: TagMeta
        :return: True if comparable objects are equal, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Let's create 2 identical TagMetas
            meta_lemon_1 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            meta_lemon_2 = sly.TagMeta('Lemon', sly.TagValueType.NONE)

            # and 1 different TagMeta and compare them to each other
            meta_cucumber = sly.TagMeta('Cucumber', sly.TagValueType.ANY_STRING)

            # Compare identical TagMetas
            meta_lemon_1 == meta_lemon_2      # True

            # Compare unidentical TagMetas
            meta_lemon_1 == meta_cucumber     # False
        """
        # TODO compare colors also here (need to check the usages and replace with is_compatible() where appropriate).
        return (
            isinstance(other, TagMeta)
            and self.name == other.name
            and self.value_type == other.value_type
            and self.possible_values == other.possible_values
        )

    def __ne__(self, other: TagMeta) -> bool:
        """
        Checks that 2 TagMetas are opposite.

        :param other: TagMeta object.
        :type other: TagMeta
        :return: True if comparable objects are not equal, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Let's create 2 identical TagMetas
            meta_lemon_1 = sly.TagMeta('Lemon', sly.TagValueType.NONE)
            meta_lemon_2 = sly.TagMeta('Lemon', sly.TagValueType.NONE)

            # and 1 different TagMeta and compare them to each other
            meta_cucumber = sly.TagMeta('Cucumber', sly.TagValueType.ANY_STRING)

            # Compare identical TagMetas
            meta_lemon_1 != meta_lemon_2      # False

            # Compare unidentical TagMetas
            meta_lemon_1 != meta_cucumber     # True
        """
        return not self == other

    def is_compatible(self, other: TagMeta) -> bool:
        """is_compatible"""
        return (
            isinstance(other, TagMeta)
            and self.name == other.name
            and self.value_type == other.value_type
            and self.possible_values == other.possible_values
        )

    def clone(
        self,
        name: Optional[str] = None,
        value_type: Optional[str] = None,
        possible_values: Optional[List[str]] = None,
        color: Optional[List[int, int, int]] = None,
        sly_id: Optional[int] = None,
        hotkey: Optional[str] = None,
        applicable_to: Optional[str] = None,
        applicable_classes: Optional[List[str]] = None,
    ) -> TagMeta:
        """
        Clone makes a copy of TagMeta with new fields, if fields are given, otherwise it will use original TagMeta fields.

        :param name: Tag name.
        :type name: str
        :param value_type: Tag value type.
        :type value_type: str
        :param possible_values: List of possible values.
        :type possible_values: List[str], optional
        :param color: :class:`[R, G, B]` color, generates random color by default.
        :type color: List[int, int, int], optional
        :param sly_id: Tag ID in Supervisely server.
        :type sly_id: int, optional
        :param hotkey: Hotkey for Tag in annotation tool UI.
        :type hotkey: str, optional
        :param applicable_to: Defines applicability of Tag only to images, objects or both.
        :type applicable_to: str, optional
        :param applicable_classes: Defines applicability of Tag only to certain classes.
        :type applicable_classes: List[str], optional
        :return: New instance of TagMeta
        :rtype: :class:`TagMeta<TagMeta>`
        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            #Original TagMeta
            meta_dog_breed = sly.TagMeta('breed', sly.TagValueType.NONE)

            # TagMetas made of original TagMeta
            # Remember that TagMeta class object is immutable, and we need to assign new instance of TagMeta to a new variable
            A_breeds = ["Affenpinscher", "Afghan Hound", "Aidi", "Airedale Terrier", "Akbash Dog", "Akita"]
            meta_A_breed = meta_dog_breed.clone(value_type=sly.TagValueType.ONEOF_STRING, possible_values=A_breeds, hotkey='A')

            B_breeds = ["Basset Fauve de Bretagne", "Basset Hound", "Bavarian Mountain Hound", "Beagle", "Beagle-Harrier", "Bearded Collie"]
            meta_B_breed = meta_A_breed.clone(possible_values=B_breeds, hotkey='B')

            C_breeds = ["Cairn Terrier", "Canaan Dog", "Canadian Eskimo Dog", "Cane Corso", "Cardigan Welsh Corgi", "Carolina Dog"]
            meta_C_breed = meta_B_breed.clone(possible_values=C_breeds, hotkey='C')
        """
        return TagMeta(
            name=take_with_default(name, self.name),
            value_type=take_with_default(value_type, self.value_type),
            possible_values=take_with_default(possible_values, self.possible_values),
            color=take_with_default(color, self.color),
            sly_id=take_with_default(sly_id, self.sly_id),
            hotkey=take_with_default(hotkey, self.hotkey),
            applicable_to=take_with_default(applicable_to, self.applicable_to),
            applicable_classes=take_with_default(applicable_classes, self.applicable_classes),
        )

    def __str__(self):
        return (
            "{:<7s}{:<24} {:<7s}{:<13} {:<13s}{:<10} {:<13s}{:<10} {:<13s}{:<10} {:<13s}{}".format(
                "Name:",
                self.name,
                "Value type:",
                self.value_type,
                "Possible values:",
                str(self.possible_values),
                "Hotkey",
                self.hotkey,
                "Applicable to",
                self.applicable_to,
                "Applicable classes",
                self.applicable_classes,
            )
        )

    @classmethod
    def get_header_ptable(cls):
        """get_header_ptable"""
        return [
            "Name",
            "Value type",
            "Possible values",
            "Hotkey",
            "Applicable to",
            "Applicable classes",
        ]

    def get_row_ptable(self):
        """get_row_ptable"""
        return [
            self.name,
            self.value_type,
            self.possible_values,
            self.hotkey,
            self.applicable_to,
            self.applicable_classes,
        ]

    def _set_id(self, id: int):
        self._sly_id = id
