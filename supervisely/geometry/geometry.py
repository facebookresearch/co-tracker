# coding: utf-8

from copy import deepcopy

import numpy as np
from supervisely.io.json import JsonSerializable
from supervisely.geometry.constants import (
    ANY_SHAPE,
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    ID,
    CLASS_ID,
)
from supervisely import logger

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

if not hasattr(np, "bool"):
    np.bool = np.bool_


# @TODO: use properties instead of field if it makes sense
class Geometry(JsonSerializable):
    """ """

    def __init__(
        self, sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None
    ):
        self.sly_id = sly_id
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at
        self.class_id = class_id

    def _add_creation_info(self, d):
        """ """
        if self.labeler_login is not None:
            d[LABELER_LOGIN] = self.labeler_login
        if self.updated_at is not None:
            d[UPDATED_AT] = self.updated_at
        if self.created_at is not None:
            d[CREATED_AT] = self.created_at
        if self.sly_id is not None:
            d[ID] = self.sly_id
        if self.class_id is not None:
            d[CLASS_ID] = self.class_id

    def _copy_creation_info_inplace(self, g):
        """ """
        self.labeler_login = g.labeler_login
        self.updated_at = g.updated_at
        self.created_at = g.created_at
        self.sly_id = g.sly_id

    @staticmethod
    def geometry_name():
        """
        :return: string with name of geometry
        """
        raise NotImplementedError()

    @classmethod
    def name(cls):
        """
        Same as geometry_name(), but shorter. In order to make the code more concise.

        :return: string with name of geometry
        """
        return cls.geometry_name()

    def crop(self, rect):
        """
        :param rect: Rectangle
        :return: list of Geometry
        """
        raise NotImplementedError()

    def relative_crop(self, rect):
        """Crops object like "crop" method, but return results with coordinates relative to rect
        :param rect:
        :return: list of Geometry
        """
        return [geom.translate(drow=-rect.top, dcol=-rect.left) for geom in self.crop(rect)]

    def rotate(self, rotator):
        """Rotates around image center -> New Geometry
        :param rotator: ImageRotator
        :return: Geometry
        """
        raise NotImplementedError()

    def resize(self, in_size, out_size):
        """
        :param in_size: (rows, cols)
        :param out_size:
            (128, 256)
            (128, KEEP_ASPECT_RATIO)
            (KEEP_ASPECT_RATIO, 256)
        :return: Geometry
        """
        raise NotImplementedError()

    def scale(self, factor):
        """Scales around origin with a given factor.
        :param: factor (float):
        :return: Geometry
        """
        raise NotImplementedError()

    def translate(self, drow, dcol):
        """
        :param drow: int rows shift
        :param dcol: int cols shift
        :return: Geometry
        """
        raise NotImplementedError()

    def fliplr(self, img_size):
        """
        :param img_size: (rows, cols)
        :return: Geometry
        """
        raise NotImplementedError()

    def flipud(self, img_size):
        """
        :param img_size: (rows, cols)
        :return: Geometry
        """
        raise NotImplementedError()

    def _draw_bool_compatible(self, draw_fn, bitmap, color, thickness, config=None):
        """ """
        if bitmap.dtype == np.bool:
            # Cannot draw on the canvas directly, create a temporary with different type.
            temp_bitmap = np.zeros(bitmap.shape[:2], dtype=np.uint8)
            draw_fn(temp_bitmap, 1, thickness=thickness, config=config)
            bitmap[temp_bitmap == 1] = color
        else:
            # Pass through the canvas without temp bitmap for efficiency.
            draw_fn(bitmap, color, thickness=thickness, config=config)

    def draw(self, bitmap, color, thickness=1, config=None):
        """
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: used only in Polyline and Point
        :param config: drawing config specific to a concrete subclass, e.g. per edge colors
        """
        self._draw_bool_compatible(self._draw_impl, bitmap, color, thickness, config)

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: used only in Polyline and Point
        """
        raise NotImplementedError()

    def draw_contour(self, bitmap, color, thickness=1, config=None):
        """Draws the figure contour on a given bitmap canvas
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: (int)
        :param config: drawing config specific to a concrete subclass, e.g. per edge colors
        """
        self._draw_bool_compatible(self._draw_contour_impl, bitmap, color, thickness, config)

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        """Draws the figure contour on a given bitmap canvas
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: (int)
        """
        raise NotImplementedError()

    @property
    def area(self):
        """
        :return: float
        """
        raise NotImplementedError()

    def to_bbox(self):
        """
        :return: Rectangle
        """
        raise NotImplementedError()

    def clone(self):
        """Clone from GEOMETRYYY"""
        return deepcopy(self)

    def validate(self, obj_class_shape, settings):
        """ """
        if obj_class_shape != ANY_SHAPE:
            if self.geometry_name() != obj_class_shape:
                raise ValueError("Geometry validation error: shape names are mismatched!")

    @staticmethod
    def config_from_json(config):
        """ """
        return config

    @staticmethod
    def config_to_json(config):
        """ """
        return config

    @classmethod
    def allowed_transforms(cls):
        """ """
        # raise NotImplementedError("{!r}".format(cls.geometry_name()))
        return []

    def convert(self, new_geometry, contour_radius=0, approx_epsilon=None):
        """ """
        from supervisely.geometry.any_geometry import AnyGeometry

        if type(self) == new_geometry or new_geometry == AnyGeometry:
            return [self]

        allowed_transforms = self.allowed_transforms()
        if new_geometry not in allowed_transforms:
            raise NotImplementedError(
                "from {!r} to {!r}".format(self.geometry_name(), new_geometry.geometry_name())
            )

        from supervisely.geometry.bitmap import Bitmap
        from supervisely.geometry.rectangle import Rectangle
        from supervisely.geometry.polygon import Polygon
        from supervisely.geometry.helpers import geometry_to_bitmap, geometry_to_polygon

        res = []
        if new_geometry == Bitmap:
            res = geometry_to_bitmap(self, radius=contour_radius)
        elif new_geometry == Rectangle:
            res = [self.to_bbox()]
        elif new_geometry == Polygon:
            res = geometry_to_polygon(self, approx_epsilon=approx_epsilon)

        if len(res) == 0:
            logger.warn(
                "Can not convert geometry {} to {} because geometry to convert is very small".format(
                    self.geometry_name(), new_geometry.geometry_name()
                )
            )
        return res
