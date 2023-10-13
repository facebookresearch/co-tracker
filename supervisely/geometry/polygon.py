# coding: utf-8


# docs
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.rectangle import Rectangle

from shapely.geometry import mapping, Polygon as ShapelyPolygon

from supervisely.geometry.conversions import shapely_figure_to_coords_list
from supervisely.geometry.point_location import (
    row_col_list_to_points,
    points_to_row_col_list,
)
from supervisely.geometry.vector_geometry import VectorGeometry
from supervisely.geometry.constants import (
    EXTERIOR,
    INTERIOR,
    POINTS,
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    ID,
    CLASS_ID,
)
from supervisely.geometry import validation
from supervisely.sly_logger import logger


class Polygon(VectorGeometry):
    """
    Polygon geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Polygon<Polygon>` class object is immutable.

    :param exterior: Exterior coordinates, object contour is defined with these points.
    :type exterior: List[PointLocation], List[List[int, int]], List[Tuple[int, int]
    :param interior: Interior coordinates, object holes are defined with these points.
    :type interior: List[List[PointLocation]], List[List[List[int, int]]], List[List[Tuple[int, int]]]
    :param sly_id: Polygon ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Polygon belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Polygon.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Polygon was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Polygon was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`, if len(exterior) < 3 or len(any element in interior list) < 3

    :Usage example:

     .. code-block:: python

            import supervisely as sly

            exterior = [sly.PointLocation(730, 2104), sly.PointLocation(2479, 402), sly.PointLocation(3746, 1646)]
            # or exterior = [[730, 2104], [2479, 402], [3746, 1646]]
            # or exterior = [(730, 2104), (2479, 402), (3746, 1646)]
            interior = [[sly.PointLocation(1907, 1255), sly.PointLocation(2468, 875), sly.PointLocation(2679, 1577)]]
            # or interior = [[[730, 2104], [2479, 402], [3746, 1646]]]
            # or interior = [[(730, 2104), (2479, 402), (3746, 1646)]]
            figure = sly.Polygon(exterior, interior)
    """

    @staticmethod
    def geometry_name():
        """
        """
        return "polygon"

    def __init__(
            self,
            exterior: Union[
                List[PointLocation], List[List[int, int]], List[Tuple[int, int]]
            ],
            interior: Union[
                List[List[PointLocation]], List[List[List[int, int]]], List[List[Tuple[int, int]]]
            ] = [],
            sly_id: Optional[int] = None,
            class_id: Optional[int] = None,
            labeler_login: Optional[int] = None,
            updated_at: Optional[str] = None,
            created_at: Optional[str] = None,
    ):
        if len(exterior) < 3:
            exterior.extend([exterior[-1]] * (3 - len(exterior)))
            logger.warn(f'"{EXTERIOR}" field must contain at least 3 points to create "Polygon" object.')
            # raise ValueError('"{}" field must contain at least 3 points to create "Polygon" object.'.format(EXTERIOR))
        for element in interior:
            if len(element) < 3:
                logger.warn(f'"{element}" interior field must contain at least 3 points to create "Polygon" object.')
                element.extend([element[-1]] * (3 - len(element)))
        # if any(len(element) < 3 for element in interior):
        #    raise ValueError('"{}" element must contain at least 3 points.'.format(INTERIOR))

        super().__init__(
            exterior,
            interior,
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    @classmethod
    def from_json(cls, data: Dict) -> Polygon:
        """
        Convert a json dict to Polygon. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Polygon in json format as a dict.
        :type data: dict
        :return: Polygon object
        :rtype: :class:`Polygon<Polygon>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_json =  {
                "points": {
                    "exterior": [
                        [2104, 730],
                        [402, 2479],
                        [1646, 3746]
                    ],
                    "interior": [
                        [
                            [1255, 1907],
                            [875, 2468],
                            [577, 2679]
                        ]
                    ]
                }
            }

            figure = sly.Polygon.from_json(figure_json)
        """
        validation.validate_geometry_points_fields(data)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(
            exterior=row_col_list_to_points(
                data[POINTS][EXTERIOR], flip_row_col_order=True
            ),
            interior=[
                row_col_list_to_points(i, flip_row_col_order=True)
                for i in data[POINTS][INTERIOR]
            ],
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def crop(self, rect: Rectangle) -> List[Polygon]:
        """
        Crops current Polygon.

        :param rect: Rectangle object for crop.
        :type rect: Rectangle
        :return: List of Polygon objects
        :rtype: :class:`List[Polygon]<Polygon>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            crop_figures = figure.crop(sly.Rectangle(1, 1, 300, 350))
        """
        try:
            # points = [
            #     PointLocation(row=rect.top, col=rect.left),
            #     PointLocation(row=rect.top, col=rect.right + 1),
            #     PointLocation(row=rect.bottom + 1, col=rect.right + 1),
            #     PointLocation(row=rect.bottom + 1, col=rect.left)
            # ]
            points = [
                PointLocation(row=rect.top, col=rect.left),
                PointLocation(row=rect.top, col=rect.right),
                PointLocation(row=rect.bottom, col=rect.right),
                PointLocation(row=rect.bottom, col=rect.left),
            ]
            # points = rect.corners # old implementation with 1 pixel error (right bottom)
            # #@TODO: investigate here (critical issue)

            clipping_window_shpl = ShapelyPolygon(points_to_row_col_list(points))
            self_shpl = ShapelyPolygon(self.exterior_np, holes=self.interior_np)
            intersections_shpl = self_shpl.buffer(0).intersection(clipping_window_shpl)
            mapping_shpl = mapping(intersections_shpl)
        except Exception:
            logger.warn("Polygon cropping exception, shapely.", exc_info=True)
            # raise
            # if polygon is invalid, just print warning and skip it
            # @TODO: need more investigation here
            return []

        intersections = shapely_figure_to_coords_list(mapping_shpl)

        # Check for bad cropping cases (e.g. empty points list)
        out_polygons = []
        for intersection in intersections:
            if (
                    isinstance(intersection, list)
                    and len(intersection) > 0
                    and len(intersection[0]) >= 3
            ):
                exterior = row_col_list_to_points(intersection[0], do_round=True)
                interiors = []
                for interior_contour in intersection[1:]:
                    if len(interior_contour) > 2:
                        interiors.append(
                            row_col_list_to_points(interior_contour, do_round=True)
                        )
                out_polygons.append(Polygon(exterior, interiors))
        return out_polygons

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """
        """
        exterior = self.exterior_np[:, ::-1]
        interior = [x[:, ::-1] for x in self.interior_np]
        bmp_to_draw = np.zeros(bitmap.shape[:2], np.uint8)
        cv2.fillPoly(bmp_to_draw, pts=[exterior], color=1)
        cv2.fillPoly(bmp_to_draw, pts=interior, color=0)
        bool_mask = bmp_to_draw.astype(bool)
        bitmap[bool_mask] = color

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        """
        """
        exterior = self.exterior_np[:, ::-1]
        interior = [x[:, ::-1] for x in self.interior_np]

        poly_lines = [exterior] + interior
        cv2.polylines(
            bitmap, pts=poly_lines, isClosed=True, color=color, thickness=thickness
        )

    # @TODO: extend possibilities, consider interior
    # returns area of exterior figure only
    @property
    def area(self) -> float:
        """
        Polygon area.

        :return: Area of current Polygon object (exterior figure only).
        :rtype: :class:`float`

        :Usage Example:

         .. code-block:: python

            print(figure.area)
            # Output: 7288.0
        """
        exterior = self.exterior_np
        return self._get_area_by_gauss_formula(exterior[:, 0], exterior[:, 1])

    @staticmethod
    def _get_area_by_gauss_formula(rows, cols):
        """
        """
        return 0.5 * np.abs(
            np.dot(rows, np.roll(cols, 1)) - np.dot(cols, np.roll(rows, 1))
        )

    def approx_dp(self, epsilon: float) -> Polygon:
        """
        Approximates a Polygon curve with the specified precision.

        :param epsilon: Specifying the approximation accuracy.
        :type epsilon: float
        :return: Polygon object
        :rtype: :class:`Polygon<Polygon>`

        :Usage Example:

         .. code-block:: python

            # Remember that Polygon class object is immutable, and we need to assign new instance of Polygon to a new variable
            approx_figure = figure.approx_dp(0.75)
        """
        exterior_np = self._approx_ring_dp(
            self.exterior_np, epsilon, closed=True
        ).tolist()
        interior_np = [
            self._approx_ring_dp(x, epsilon, closed=True).tolist()
            for x in self.interior_np
        ]
        exterior = row_col_list_to_points(exterior_np, do_round=True)
        interior = [row_col_list_to_points(x, do_round=True) for x in interior_np]
        return Polygon(exterior, interior)

    @classmethod
    def allowed_transforms(cls):
        """
        """
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.rectangle import Rectangle
        from supervisely.geometry.bitmap import Bitmap

        return [AnyGeometry, Rectangle, Bitmap]
