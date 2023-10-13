# coding: utf-8

# docs
from __future__ import annotations
from copy import deepcopy
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Iterable
from supervisely.geometry.image_rotator import ImageRotator

from supervisely.geometry.constants import (
    EXTERIOR,
    INTERIOR,
    POINTS,
    GEOMETRY_SHAPE,
    GEOMETRY_TYPE,
)
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.point_location import PointLocation, points_to_row_col_list
from supervisely.geometry.rectangle import Rectangle


class VectorGeometry(Geometry):
    """
    VectorGeometry is a base class of geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`VectorGeometry<VectorGeometry>` class object is immutable.

    :param exterior: Exterior coordinates, object contour is defined with these points (used for :class:`Polygon<supervisely.geometry.polygon.Polygon>`).
    :type exterior: List[PointLocation], List[List[int, int]], List[Tuple[int, int]
    :param interior: Interior coordinates, object holes is defined with these points (used for :class:`Polygon<supervisely.geometry.polygon.Polygon>`).
    :type interior: List[List[PointLocation]], List[List[List[int, int]]], List[List[Tuple[int, int]]]
    :param sly_id: VectorGeometry ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which VectorGeometry belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created VectorGeometry.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when VectorGeometry was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when VectorGeometry was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`TypeError`, if exterior or interior parameters are not a list of PointLocation objects

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

    def __init__(
            self,
            exterior: Union[
                List[PointLocation], List[List[int, int]], List[Tuple[int, int]]
            ],
            interior: Union[
                List[List[PointLocation]],
                List[List[List[int, int]]],
                List[List[Tuple[int, int]]],
            ] = [],
            sly_id: Optional[int] = None,
            class_id: Optional[int] = None,
            labeler_login: Optional[int] = None,
            updated_at: Optional[str] = None,
            created_at: Optional[str] = None,
    ):
        result_exterior = []
        if not isinstance(exterior, list):
            raise TypeError('Argument "exterior" must be a list of coordinates')
        for p in exterior:
            if isinstance(p, PointLocation):
                result_exterior.append(p)
            elif isinstance(p, tuple) and len(p) == 2:
                result_exterior.append(PointLocation(p[0], p[1]))
            elif isinstance(p, list) and len(p) == 2:
                result_exterior.append(PointLocation(p[0], p[1]))
            else:
                raise TypeError(
                    'Type of items (coordinates) in list "exterior" have to be tuple(int, int) or list[int, int] or PointLocation(row, col)'
                )

        result_interior = []
        if not isinstance(interior, list):
            raise TypeError(
                'Argument "interior" must be a list of lists with coordinates'
            )
        for coords in interior:
            if not isinstance(interior, list):
                raise TypeError('"interior" coords must be a list of coordinates')
            p_coords = []
            for p in coords:
                if isinstance(p, PointLocation):
                    p_coords.append(p)
                elif isinstance(p, tuple) and len(p) == 2:
                    p_coords.append(PointLocation(p[0], p[1]))
                elif isinstance(p, list) and len(p) == 2:
                    p_coords.append(PointLocation(p[0], p[1]))
                else:
                    raise TypeError(
                        'Type of items (coordinates) in list "interior" have to be tuple(int, int) or list[int, int] or PointLocation(row, col)'
                    )
            result_interior.append(p_coords)

        self._exterior = deepcopy(result_exterior)
        self._interior = deepcopy(result_interior)
        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def to_json(self) -> Dict:
        """
        Convert the VectorGeometry to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            figure_json = figure.to_json()
            print(figure_json)
            # Output: {
            #    "points": {
            #        "exterior": [
            #            [2104, 730],
            #            [402, 2479],
            #            [1646, 3746]
            #        ],
            #        "interior": [
            #            [
            #                [1255, 1907],
            #                [875, 2468],
            #                [1577, 2679]
            #            ]
            #        ]
            #    }
            # }
        """
        packed_obj = {
            POINTS: {
                EXTERIOR: points_to_row_col_list(
                    self._exterior, flip_row_col_order=True
                ),
                INTERIOR: [
                    points_to_row_col_list(i, flip_row_col_order=True)
                    for i in self._interior
                ],
            },
            GEOMETRY_SHAPE: self.geometry_name(),
            GEOMETRY_TYPE: self.geometry_name(),
        }
        self._add_creation_info(packed_obj)
        return packed_obj

    @property
    def exterior(self) -> List[PointLocation]:
        """
        VectorGeometry exterior points.

        :return: VectorGeometry exterior points
        :rtype: :class:`List[PointLocation]<supervisely.geometry.point_location.PointLocation>`
        :Usage example:

         .. code-block:: python

            exterior = figure.exterior
        """
        return deepcopy(self._exterior)

    @property
    def exterior_np(self) -> np.ndarray:
        """
        Converts exterior attribute of VectorGeometry to numpy array.

        :return: Numpy array
        :rtype: :class:`np.ndarray`

        :Usage example:

         .. code-block:: python

            print(figure.exterior_np)
            # Output:
            # [[ 730 2104]
            #  [2479  402]
            #  [3746 1646]]
        """
        return np.array(points_to_row_col_list(self._exterior), dtype=np.int64)

    @property
    def interior(self) -> List[List[PointLocation]]:
        """
        VectorGeometry interior points.

        :return: VectorGeometry interior points
        :rtype: :class:`List[List[PointLocation]]<supervisely.geometry.point_location.PointLocation>`
        :Usage example:

         .. code-block:: python

            interior = figure.interior
        """
        return deepcopy(self._interior)

    @property
    def interior_np(self):
        """
        Converts interior attribute of VectorGeometry to numpy array.

        :return: Numpy array
        :rtype: :class:`List[np.ndarray]`

        :Usage example:

         .. code-block:: python

            print(figure.interior_np)
            # Output:
            # [array([[1907, 1255],
            #        [2468,  875],
            #        [2679, 1577]])]
        """
        return [
            np.array(points_to_row_col_list(i), dtype=np.int64) for i in self._interior
        ]

    def _transform(self, transform_fn):
        """
        """
        result = deepcopy(self)
        result._exterior = [transform_fn(p) for p in self._exterior]
        result._interior = [[transform_fn(p) for p in i] for i in self._interior]
        return result

    def resize(
            self, in_size: Tuple[int, int], out_size: Tuple[int, int]
    ) -> VectorGeometry:
        """
        Resizes current VectorGeometry.

        :param in_size: Input image size (height, width) to which belongs VectorGeometry.
        :type in_size: Tuple[int, int]
        :param out_size: Desired output image size (height, width) to which belongs VectorGeometry.
        :type out_size: Tuple[int, int]
        :return: VectorGeometry object
        :rtype: :class:`VectorGeometry<VectorGeometry>`

        :Usage Example:

         .. code-block:: python

            # Remember that VectorGeometry class object is immutable, and we need to assign new instance of VectorGeometry to a new variable
            in_height, in_width = 300, 400
            out_height, out_width = 600, 800
            resize_figure = figure.resize((in_height, in_width), (out_height, out_width))
        """
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor: float) -> VectorGeometry:
        """
        Scales current VectorGeometry.

        :param factor: Scale parameter.
        :type factor: float
        :return: VectorGeometry object
        :rtype: :class:`VectorGeometry<VectorGeometry>`

        :Usage Example:

         .. code-block:: python

            # Remember that VectorGeometry class object is immutable, and we need to assign new instance of VectorGeometry to a new variable
            scale_figure = figure.scale(0.75)
        """
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow: int, dcol: int) -> VectorGeometry:
        """
        Translates current VectorGeometry.

        :param drow: Horizontal shift.
        :type drow: int
        :param dcol: Vertical shift.
        :type dcol: int
        :return: VectorGeometry object
        :rtype: :class:`VectorGeometry<VectorGeometry>`

        :Usage Example:

         .. code-block:: python

            # Remember that VectorGeometry class object is immutable, and we need to assign new instance of VectorGeometry to a new variable
            translate_figure = figure.translate(150, 250)
        """
        return self._transform(lambda p: p.translate(drow, dcol))

    def rotate(self, rotator: ImageRotator) -> VectorGeometry:
        """
        Rotates current VectorGeometry.

        :param rotator: ImageRotator object for rotate.
        :type rotator: ImageRotator
        :return: VectorGeometry object
        :rtype: :class:`VectorGeometry<VectorGeometry>`

        :Usage Example:

         .. code-block:: python

            from supervisely.geometry.image_rotator import ImageRotator

            # Remember that VectorGeometry class object is immutable, and we need to assign new instance of VectorGeometry to a new variable
            height, width = 300, 400
            rotator = ImageRotator((height, width), 25)
            rotate_figure = figure.rotate(rotator)

        """
        return self._transform(lambda p: p.rotate(rotator))

    def fliplr(self, img_size: Tuple[int, int]) -> VectorGeometry:
        """
        Flips current VectorGeometry in horizontal.

        :param img_size: Input image size (height, width) to which belongs VectorGeometry.
        :type img_size: Tuple[int, int]
        :return: VectorGeometry object
        :rtype: :class:`VectorGeometry<VectorGeometry>`

        :Usage Example:

         .. code-block:: python

            # Remember that VectorGeometry class object is immutable, and we need to assign new instance of VectorGeometry to a new variable
            height, width = 300, 400
            fliplr_figure = figure.fliplr((height, width))
        """
        return self._transform(lambda p: p.fliplr(img_size))

    def flipud(self, img_size: Tuple[int, int]) -> VectorGeometry:
        """
        Flips current VectorGeometry in vertical.

        :param img_size: Input image size (height, width) to which belongs VectorGeometry.
        :type img_size: Tuple[int, int]
        :return: VectorGeometry object
        :rtype: :class:`VectorGeometry<VectorGeometry>`

        :Usage Example:

         .. code-block:: python

            # Remember that VectorGeometry class object is immutable, and we need to assign new instance of VectorGeometry to a new variable
            height, width = 300, 400
            flipud_figure = figure.flipud((height, width))
        """
        return self._transform(lambda p: p.flipud(img_size))

    def to_bbox(self) -> Rectangle:
        """
        Creates Rectangle object from current VectorGeometry.

        :return: Rectangle object
        :rtype: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`

        :Usage Example:

         .. code-block:: python

            rectangle = figure.to_bbox()
        """
        exterior_np = self.exterior_np
        rows, cols = exterior_np[:, 0], exterior_np[:, 1]
        return Rectangle(
            top=round(min(rows).item()),
            left=round(min(cols).item()),
            bottom=round(max(rows).item()),
            right=round(max(cols).item()),
        )

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: used only in Polyline and Point
        """
        self._draw_contour_impl(bitmap, color, thickness, config=config)

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        """Draws the figure contour on a given bitmap canvas
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: (int)
        """
        raise NotImplementedError()

    @staticmethod
    def _approx_ring_dp(ring, epsilon, closed):
        """
        """
        new_ring = cv2.approxPolyDP(ring.astype(np.int32), epsilon, closed)
        new_ring = np.squeeze(new_ring, 1)
        if len(new_ring) < 3 and closed:
            new_ring = ring.astype(np.int32)
        return new_ring

    def approx_dp(self, epsilon):
        """
        """
        raise NotImplementedError()
