# coding: utf-8


# docs

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from supervisely.geometry.point import PointLocation
from supervisely.geometry.rectangle import Rectangle

from shapely.geometry import mapping, LineString, Polygon as ShapelyPolygon
from supervisely.geometry.conversions import shapely_figure_to_coords_list
from supervisely.geometry.point_location import row_col_list_to_points
from supervisely.geometry.vector_geometry import VectorGeometry
from supervisely.geometry.constants import (
    EXTERIOR,
    POINTS,
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    ID,
    CLASS_ID,
)
from supervisely.geometry import validation
from supervisely import logger


class Polyline(VectorGeometry):
    """
    Polyline geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Polyline<Polyline>` class object is immutable.

    :param exterior: List of exterior coordinates, the Polyline is defined with these points.
    :type exterior: List[PointLocation], List[List[int, int]], List[Tuple[int, int]
    :param sly_id: Polyline ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Polyline belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Polyline.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Polyline was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Polyline was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`, field exterior must contain at least two points to create Polyline object

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        exterior = [sly.PointLocation(730, 2104), sly.PointLocation(2479, 402), sly.PointLocation(1500, 780)]
        # or exterior = [[730, 2104], [2479, 402], [1500, 780]]
        # or exterior = [(730, 2104), (2479, 402), (1500, 780)]
        figure = sly.Polyline(exterior)
    """

    @staticmethod
    def geometry_name():
        """
        """
        return "line"

    def __init__(
        self,
        exterior: Union[
            List[PointLocation], List[List[int, int]], List[Tuple[int, int]]
        ],
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[int] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        if len(exterior) < 2:
            raise ValueError(f'"{EXTERIOR}" field must contain at least two points to create "Polyline" object.')

        super().__init__(
            exterior=exterior,
            interior=[],
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    @classmethod
    def from_json(cls, data: Dict) -> Polyline:
        """
        Convert a json dict to Polyline. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Polyline in json format as a dict.
        :type data: dict
        :return: Polyline object
        :rtype: :class:`Polyline<Polyline>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_json = {
                "points": {
                    "exterior": [
                        [2104, 730],
                        [402, 2479],
                        [780, 1500]
                    ],
                    "interior": []
                }
            }
            figure = sly.Polyline.from_json(figure_json)
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
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def crop(self, rect: Rectangle) -> List[Polyline]:
        """
        Crops current Polyline.

        :param rect: Rectangle object for crop.
        :type rect: Rectangle
        :return: List of Polyline objects
        :rtype: :class:`List[Polyline]<Polyline>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            crop_figures = figure.crop(sly.Rectangle(0, 0, 100, 200))
        """
        try:
            clipping_window = [
                [rect.top, rect.left],
                [rect.top, rect.right],
                [rect.bottom, rect.right],
                [rect.bottom, rect.left],
            ]
            clipping_window_shpl = ShapelyPolygon(clipping_window)

            exterior = self.exterior_np
            intersections_polygon = LineString(exterior).intersection(
                clipping_window_shpl
            )
            mapping_shpl = mapping(intersections_polygon)
        except Exception:
            logger.warn("Line cropping exception, shapely.", exc_info=False)
            raise

        res_lines_pts = shapely_figure_to_coords_list(mapping_shpl)

        # tiny hack to combine consecutive segments
        lines_combined = []
        if res_lines_pts != [()]:
            for simple_l in res_lines_pts:
                if len(lines_combined) > 0:
                    prev = lines_combined[-1]
                    if prev[-1] == simple_l[0]:
                        lines_combined[-1] = list(prev) + list(simple_l[1:])
                        continue
                lines_combined.append(simple_l)

        return [Polyline(row_col_list_to_points(line)) for line in lines_combined]

    def _draw_impl(self, bitmap: np.ndarray, color, thickness=1, config=None):
        """
        """
        self._draw_contour_impl(bitmap, color, thickness, config=config)

    def _draw_contour_impl(self, bitmap: np.ndarray, color, thickness=1, config=None):
        """
        """
        exterior = self.exterior_np[:, ::-1]
        cv2.polylines(
            bitmap, pts=[exterior], isClosed=False, color=color, thickness=thickness
        )

    @property
    def area(self) -> float:
        """
        Polyline area, always 0.0.

        :return: Area of current Polyline
        :rtype: :class:`float`

        :Usage Example:

         .. code-block:: python

            print(figure.area)
            # Output: 0.0
        """
        return 0.0

    def approx_dp(self, epsilon: float) -> Polyline:
        """
        Approximates a Polyline with the specified precision.

        :param epsilon: Specifying the approximation accuracy.
        :type epsilon: float
        :return: Polyline object
        :rtype: :class:`Polyline<Polyline>`

        :Usage Example:

         .. code-block:: python

            # Remember that Polyline class object is immutable, and we need to assign new instance of Polyline to a new variable
            approx_figure = figure.approx_dp(0.75)
        """
        exterior_np = self._approx_ring_dp(
            self.exterior_np, epsilon, closed=True
        ).tolist()
        exterior = row_col_list_to_points(exterior_np, do_round=True)
        return Polyline(exterior)

    @classmethod
    def allowed_transforms(cls):
        """
        """
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.rectangle import Rectangle
        from supervisely.geometry.bitmap import Bitmap
        from supervisely.geometry.polygon import Polygon

        return [AnyGeometry, Rectangle, Bitmap, Polygon]
