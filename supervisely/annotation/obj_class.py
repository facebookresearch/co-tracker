# coding: utf-8
"""General information about :class:`Label<supervisely.annotation.label.LabelBase>`"""

# docs
from __future__ import annotations
from copy import deepcopy
from typing import List, Optional, Dict, Union

from supervisely.imaging.color import random_rgb, rgb2hex, hex2rgb, _validate_color
from supervisely.io.json import JsonSerializable
from supervisely.collection.key_indexed_collection import KeyObject
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely._utils import take_with_default
from supervisely.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely.geometry.graph import GraphNodes, KeypointsTemplate


class ObjClassJsonFields:
    """Json fields for :class:`Annotation<supervisely.annotation.obj_class.ObjClass>`"""

    ID = "id"
    """"""
    NAME = "title"
    """"""
    GEOMETRY_TYPE = "shape"
    """"""
    COLOR = "color"
    """"""
    GEOMETRY_CONFIG = "geometry_config"
    """"""
    HOTKEY = "hotkey"
    """"""


class ObjClass(KeyObject, JsonSerializable):
    """
    General information about :class:`Label<supervisely.annotation.label.Label>`. :class:`ObjClass` object is immutable.

    :param name: Class name.
    :type name: str
    :param geometry_type: Defines the shape of ObjClass: :class:`Bitmap<supervisely.geometry.bitmap.Bitmap>`, :class:`Cuboid<supervisely.geometry.cuboid.Cuboid>`, :class:`Graph<supervisely.geometry.graph.GraphNodes>`, :class:`Point<supervisely.geometry.point.Point>`, :class:`Polygon<supervisely.geometry.polygon.Polygon>`, :class:`Polyline<supervisely.geometry.polyline.Polyline>`, :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`.
    :type geometry_type: dict, optional
    :param color: :class:`[R, G, B]`, generates random color by default.
    :type color: List[int, int, int], optional
    :param geometry_config: Additional settings of the geometry.
    :type geometry_config: dict, optional
    :param sly_id: ID in Supervisely server.
    :type sly_id: int, optional
    :param hotkey: Hotkey for ObjClass in annotation tool UI.
    :type hotkey: str, optional
    :raises: :class:`ValueError`, if color is not list or tuple, or doesn't have exactly 3 values
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Simple ObjClass example
        class_lemon = sly.ObjClass('lemon', sly.Rectangle)

        # More complex ObjClass example
        class_cucumber = sly.ObjClass('cucumber', sly.Bitmap, color=[128, 0, 255], hotkey='d')
    """

    def __init__(
        self,
        name: str,
        geometry_type: type,
        color: Optional[List[int]] = None,
        geometry_config: Optional[Union[Dict, KeypointsTemplate]] = None,
        sly_id: Optional[int] = None,
        hotkey: Optional[str] = None,
    ):
        self._name = name
        self._geometry_type = geometry_type
        self._color = random_rgb() if color is None else deepcopy(color)
        if geometry_type == GraphNodes and geometry_config is None:
            raise ValueError("sly.GraphNodes requires geometry_config to be passed to sly.ObjClass")

        if isinstance(geometry_config, KeypointsTemplate):
            geometry_config = geometry_config.config
        self._geometry_config = deepcopy(take_with_default(geometry_config, {}))
        self._sly_id = sly_id
        self._hotkey = take_with_default(hotkey, "")
        _validate_color(self._color)

    @property
    def name(self) -> str:
        """
        Name.

        :return: Name
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            print(class_lemon.name)
            # Output: 'lemon'
        """
        return self._name

    def key(self) -> str:
        """
        Used as a key in ObjClassCollection (like key in dict)

        :return: string name of the ObjectClass
        :rtype: :class:`Str`
        """
        return self.name

    @property
    def geometry_type(self) -> type:
        """
        Type of the geometry that is associated with ObjClass.

        :return: Geometry type
        :rtype: :class:`type`
        :Usage example:

         .. code-block:: python

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            print(class_lemon.geometry_type)
            # Output: <class 'supervisely.geometry.rectangle.Rectangle'>

            class_kiwi = sly.ObjClass('kiwi', sly.Bitmap)
            print(class_kiwi.geometry_type)
            # Output: <class 'supervisely.geometry.bitmap.Bitmap'>
        """
        return self._geometry_type

    @property
    def geometry_config(self) -> Dict:
        # """"""
        # SPHINX ERROR: >>> line = doc.splitlines()[0]
        # IndexError: list index out of range
        return deepcopy(self._geometry_config)

    @property
    def color(self) -> List[int, int, int]:
        """
        :class:`[R,G,B]` color.

        :return: Color
        :rtype: :class:`List[int, int, int]`
        :Usage example:

         .. code-block:: python

            class_lemon = sly.ObjClass('lemon', sly.Rectangle, color=[255,120,0])
            print(class_lemon.color)
            # Output: [255,120,0]
        """
        return deepcopy(self._color)

    @property
    def sly_id(self) -> int:
        """
        Class ID in Supervisely server.

         :return: ID
         :rtype: :class:`int`
         :Usage example:

          .. code-block:: python

             class_lemon = sly.ObjClass('lemon', sly.Rectangle, sly_id=38584)
             print(class_lemon.sly_id)
             # Output: 38584
        """
        return self._sly_id

    @property
    def hotkey(self) -> str:
        """
        Hotkey for ObjClass in annotation tool UI..

        :return: Hotkey
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            class_lemon = sly.ObjClass('lemon', sly.Rectangle, hotkey='M')
            print(class_lemon.hotkey)
            # Output: 'M'
        """
        return self._hotkey

    def to_json(self) -> Dict:
        """
        Convert the ObjClass to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            lemon_json = class_lemon.to_json()

            print(lemon_json)
            # Output: {
            #     "title": "lemon",
            #     "shape": "rectangle",
            #     "color": "#8A2F0F",
            #     "geometry_config": {},
            #     "hotkey": ""
            # }
        """
        res = {
            ObjClassJsonFields.NAME: self.name,
            ObjClassJsonFields.GEOMETRY_TYPE: self.geometry_type.geometry_name(),
            ObjClassJsonFields.COLOR: rgb2hex(self.color),
            ObjClassJsonFields.GEOMETRY_CONFIG: self.geometry_type.config_to_json(
                self._geometry_config
            ),
        }
        if self.sly_id is not None:
            res[ObjClassJsonFields.ID] = self.sly_id
        if self._hotkey is not None:
            res[ObjClassJsonFields.HOTKEY] = self.hotkey
        return res

    @classmethod
    def from_json(cls, data: Dict) -> ObjClass:
        """
        Convert a json dict to ObjClass. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: ObjClass in json format as a dict.
        :type data: dict
        :return: ObjClass object
        :rtype: :class:`ObjClass<ObjClass>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            data = {
                "title": "lemon",
                "shape": "rectangle",
                "color": "#0F6E8A",
                "hotkey": "Q"
            }

            class_lemon = sly.ObjClass.from_json(data)
        """
        name = data[ObjClassJsonFields.NAME]
        geometry_type = GET_GEOMETRY_FROM_STR(data[ObjClassJsonFields.GEOMETRY_TYPE])
        color = hex2rgb(data[ObjClassJsonFields.COLOR])
        geometry_config = geometry_type.config_from_json(
            data.get(ObjClassJsonFields.GEOMETRY_CONFIG)
        )
        sly_id = data.get(ObjClassJsonFields.ID, None)
        hotkey = data.get(ObjClassJsonFields.HOTKEY, "")
        return cls(
            name=name,
            geometry_type=geometry_type,
            color=color,
            geometry_config=geometry_config,
            sly_id=sly_id,
            hotkey=hotkey,
        )

    def __eq__(self, other: ObjClass) -> bool:
        """
        Checks that 2 ObjClass objects are equal by comparing their name, geometry type and geometry config.

        :param other: ObjClass object.
        :type other: ObjClass
        :return: True if comparable objects are equal, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Let's create 2 identical ObjClasses and 1 different ObjClass and compare them to each other
            class_lemon_1 = sly.ObjClass('Lemon', sly.Rectangle)
            class_lemon_2 = sly.ObjClass('Lemon', sly.Rectangle)
            class_cucumber = sly.ObjClass('Cucumber', sly.Rectangle)

            # Compare identical ObjClasses
            class_lemon_1 == class_lemon_2      # True

            # Compare unidentical ObjClasses
            class_lemon_1 == class_cucumber     # False
        """
        return (
            isinstance(other, ObjClass)
            and self.name == other.name
            and (
                self.geometry_type == other.geometry_type
                or AnyGeometry in [self.geometry_type, other.geometry_type]
            )
            and self.geometry_config == other.geometry_config
        )

    def __ne__(self, other: ObjClass) -> bool:
        """
        Checks that 2 ObjClass objects are opposite.

        :param other: ObjClass object.
        :type other: ObjClass
        :return: True if comparable objects are not equal, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Let's create 2 identical ObjClasses
            class_lemon_1 = sly.ObjClass('Lemon', sly.Rectangle)
            class_lemon_2 = sly.ObjClass('Lemon', sly.Rectangle)

            # and 1 different ObjClass and compare them to each other
            class_cucumber = sly.ObjClass('Cucumber', sly.Rectangle)

            # Compare identical ObjClasses
            class_lemon_1 != class_lemon_2      # False

            # Compare unidentical ObjClasses
            class_lemon_1 != class_cucumber     # True
        """
        return not self == other

    def __str__(self):  # Is need show geometry settings here?
        return "{:<7s}{:<10}{:<7s}{:<13}{:<7s}{:<15}{:<16s}{:<16}{:<7s}{:<7}".format(
            "Name:",
            self.name,
            "Shape:",
            self.geometry_type.__name__,
            "Color:",
            str(self.color),
            "Geom. settings:",
            str(self.geometry_config),
            "Hotkey",
            self.hotkey,
        )

    @classmethod
    def get_header_ptable(cls):
        """
        get_header_ptable
        """
        return ["Name", "Shape", "Color", "Hotkey"]  # Is need show geometry settings here?

    def get_row_ptable(self):
        """get_row_ptable"""
        return [self.name, self.geometry_type.__name__, self.color, self.hotkey]

    def clone(
        self,
        name: Optional[str] = None,
        geometry_type: Optional[Geometry] = None,
        color: Optional[List[int, int, int]] = None,
        geometry_config: Optional[Dict] = None,
        sly_id: Optional[int] = None,
        hotkey: Optional[str] = None,
    ) -> ObjClass:
        """
        Makes a copy of ObjClass with new fields, if fields are given, otherwise it will use fields of the original ObjClass.

        :param name: Class name.
        :type name: str
        :param geometry_type: Defines the shape of ObjClass: :class:`Bitmap<supervisely.geometry.bitmap.Bitmap>`, :class:`Cuboid<supervisely.geometry.cuboid.Cuboid>`, :class:`Point<supervisely.geometry.point.Point>`, :class:`Polygon<supervisely.geometry.polygon.Polygon>`, :class:`Polyline<supervisely.geometry.polyline.Polyline>`, :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`.
        :type geometry_type: type
        :param color: :class:`[R, G, B]`, generates random color by default.
        :type color: List[int, int, int], optional
        :param geometry_config: Additional settings of the geometry.
        :type geometry_config: dict, optional
        :param sly_id: ID in Supervisely server.
        :type sly_id: int, optional
        :param hotkey: Hotkey for ObjClass in annotation tool UI.
        :type hotkey: str, optional
        :return: New instance of ObjClass
        :rtype: :class:`ObjClass<ObjClass>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)

            # Let's clone our ObjClass, but with different name
            # Remember that ObjClass object is immutable, and we need to assign new instance of ObjClass to a new variable
            clone_lemon_1 = class_lemon.clone(name="lemon clone")

            # Let's clone our ObjClass, but with different color and hotkey
            # Remember that ObjClass object is immutable, and we need to assign new instance of ObjClass to a new variable
            clone_lemon_2 = class_lemon.clone(color=[128, 0, 64], hotkey='Q')

            #  Let's clone our ObjClass without new fields
            clone_lemon_3 = class_lemon.clone()
        """
        return ObjClass(
            name=take_with_default(name, self.name),
            geometry_type=take_with_default(geometry_type, self.geometry_type),
            color=take_with_default(color, self.color),
            geometry_config=take_with_default(geometry_config, self.geometry_config),
            sly_id=take_with_default(sly_id, self.sly_id),
            hotkey=take_with_default(hotkey, self.hotkey),
        )

    def __hash__(self):
        return hash((self.name, self.geometry_type.geometry_name()))

    def _set_id(self, id: int):
        self._sly_id = id
