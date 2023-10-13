# coding: utf-8


# docs
from __future__ import annotations
import cv2
from typing import List, Tuple, Dict, Optional
from supervisely.geometry.image_rotator import ImageRotator


from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.rectangle import Rectangle
from supervisely._utils import unwrap_if_numpy
from supervisely.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID


class Point(Geometry):
    """
    Point geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Point<Point>` object is immutable.

    :param row: Position of Point on height.
    :type row: int or float
    :param col: Position of Point on width.
    :type col: int or float
    :param sly_id: Point ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Point belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Point.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Point was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Point was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        row = 100
        col = 200
        figure = sly.Point(row, col)
    """
    def __init__(self, row: int, col: int,
                 sly_id: Optional[int] = None, class_id: Optional[int] = None, labeler_login: Optional[int] = None,
                 updated_at: Optional[str] = None, created_at: Optional[str] = None):
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
        self._row = round(unwrap_if_numpy(row))
        self._col = round(unwrap_if_numpy(col))

    @property
    def row(self) -> int:
        """
        Position of Point height.

        :return: Height of Point
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(figure.row)
            # Output: 100
        """
        return self._row

    @property
    def col(self) -> int:
        """
        Position of Point width.

        :return: Width of Point
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(figure.col)
            # Output: 200
        """
        return self._col

    @classmethod
    def from_point_location(cls, pt: PointLocation, sly_id: Optional[int] = None, class_id: Optional[int] = None,
                            labeler_login: Optional[int] = None, updated_at: Optional[str] = None, created_at: Optional[str] = None) -> Point:
        """
        Create Point from given :class:`PointLocation<supervisely.geometry.point_location.PointLocation>` object.

        :param pt: PointLocation object.
        :type pt: PointLocation
        :param sly_id: Point ID in Supervisely server.
        :type sly_id: int, optional
        :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Point belongs.
        :type class_id: int, optional
        :param labeler_login: Login of the user who created Point.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when Point was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when Point was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :return: Point object
        :rtype: :class:`Point<Point>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_loc = sly.PointLocation(100, 200)
            figure = sly.Point.from_point_location(figure_loc)
        """
        return cls(row=pt.row, col=pt.col,
                   sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    @property
    def point_location(self) -> PointLocation:
        """
        Create PointLocation object from Point.

        :return: PointLocation object
        :rtype: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`
        :Usage example:

         .. code-block:: python

            figure_loc = figure.point_location
        """
        return PointLocation(row=self.row, col=self.col)

    @staticmethod
    def geometry_name():
        """
        """
        return 'point'

    def crop(self, rect: Rectangle) -> List[Point]:
        """
        Crops current Point.

        :param rect: Rectangle object for crop.
        :type rect: Rectangle
        :return: List of Point objects
        :rtype: :class:`List[Point]<Point>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            crop_figures = figure.crop(sly.Rectangle(1, 1, 300, 350))
        """
        return [self.clone()] if rect.contains_point_location(self.point_location) else []

    def rotate(self, rotator: ImageRotator) -> Point:
        """
        Rotates current Point.

        :param rotator: ImageRotator object for rotation.
        :type rotator: ImageRotator
        :return: Point object
        :rtype: :class:`Point<Point>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.image_rotator import ImageRotator

            # Remember that Point class object is immutable, and we need to assign new instance of Point to a new variable
            height, width = 300, 400
            rotator = ImageRotator((height, width), 25)
            rotate_figure = figure.rotate(rotator)
        """
        return self.from_point_location(self.point_location.rotate(rotator))

    def resize(self, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> Point:
        """
        Resizes current Point.

        :param in_size: Input Input image size (height, width) to which belongs Point.
        :type in_size: Tuple[int, int]
        :param out_size: Desired output image size (height, width) to which belongs Point.
        :type out_size: Tuple[int, int]
        :return: Point object
        :rtype: :class:`Point<Point>`

        :Usage Example:

         .. code-block:: python

            # Remember that Point class object is immutable, and we need to assign new instance of Point to a new variable
            in_height, in_width = 300, 400
            out_height, out_width = 600, 800
            resize_figure = figure.resize((in_height, in_width), (out_height, out_width))
        """
        return self.from_point_location(self.point_location.resize(in_size, out_size))

    def fliplr(self, img_size: Tuple[int, int]) -> Point:
        """
        Flips current Point in horizontal.

        :param img_size: Input image size (height, width) to which belongs Point object.
        :type img_size: Tuple[int, int]
        :return: Point object
        :rtype: :class:`Point<Point>`

        :Usage Example:

         .. code-block:: python

            # Remember that Point class object is immutable, and we need to assign new instance of Point to a new variable
            height, width = 300, 400
            fliplr_figure = figure.fliplr((height, width))
        """
        return self.from_point_location(self.point_location.fliplr(img_size))

    def flipud(self, img_size: Tuple[int, int]) -> Point:
        """
        Flips current Point in vertical.

        :param img_size: Input image size (height, width) to which belongs Point object.
        :type img_size: Tuple[int, int]
        :return: Point object
        :rtype: :class:`Point<Point>`

        :Usage Example:

         .. code-block:: python

            # Remember that Point class object is immutable, and we need to assign new instance of Point to a new variable
            height, width = 300, 400
            flipud_figure = figure.flipud((height, width))
        """
        return self.from_point_location(self.point_location.flipud(img_size))

    def scale(self, factor: float) -> Point:
        """
        Scales current Point.

        :param factor: Scale parameter.
        :type factor: float
        :return: Point object
        :rtype: :class:`Point<Point>`

        :Usage Example:

         .. code-block:: python

            # Remember that Point class object is immutable, and we need to assign new instance of Point to a new variable
            scale_figure = figure.scale(0.75)
        """
        return self.from_point_location(self.point_location.scale(factor))

    def translate(self, drow: int, dcol: int) -> Point:
        """
        Translates current Point.

        :param drow: Horizontal shift.
        :type drow: int
        :param dcol: Vertical shift.
        :type dcol: int
        :return: Point object
        :rtype: :class:`Point<Point>`

        :Usage Example:

         .. code-block:: python

            # Remember that Point class object is immutable, and we need to assign new instance of Point to a new variable
            translate_figure = figure.translate(150, 350)
        """
        return self.from_point_location(self.point_location.translate(drow, dcol))

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        r = round(thickness / 2)  # @TODO: relation between thickness and point radius - ???
        cv2.circle(bitmap, (self.col, self.row), radius=r, color=color, thickness=cv2.FILLED)

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        # @TODO: mb dummy operation for Point
        r = round((thickness + 1) / 2)
        cv2.circle(bitmap, (self.col, self.row), radius=r, color=color, thickness=cv2.FILLED)

    @property
    def area(self) -> float:
        """
        Point area.

        :return: Area of current Point object, always 0.0
        :rtype: :class:`float`

        :Usage Example:

         .. code-block:: python

            print(figure.area)
            # Output: 0.0
        """
        return 0.0

    def to_bbox(self) -> Rectangle:
        """
        Create Rectangle object from current Point.

        :return: Rectangle object
        :rtype: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`

        :Usage Example:

         .. code-block:: python

            rectangle = figure.to_bbox()
        """
        return Rectangle(top=self.row, left=self.col, bottom=self.row, right=self.col)

    def to_json(self) -> Dict:
        """
        Convert the Point to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            figure_json = figure.to_json()
            print(figure_json)
            # Output: {
            #    "points": {
            #        "exterior": [
            #            [200, 100]
            #        ],
            #        "interior": []
            #    }
            # }
        """
        res = self.point_location.to_json()
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data: Dict) -> Point:
        """
        Convert a json dict to Point. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Point in json format as a dict.
        :type data: dict
        :return: Point object
        :rtype: :class:`Point<Point>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_json = {
                "points": {
                    "exterior": [
                        [200, 100]
                    ],
                    "interior": []
                }
            }
            figure = sly.Point.from_json(figure_json)
        """
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls.from_point_location(PointLocation.from_json(data),
                                       sly_id=sly_id, class_id=class_id,
                                       labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    @classmethod
    def allowed_transforms(cls):
        """
        """
        from supervisely.geometry.any_geometry import AnyGeometry
        return [AnyGeometry]