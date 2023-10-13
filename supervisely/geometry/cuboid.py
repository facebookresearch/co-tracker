# coding: utf-8

# docs
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point_location import PointLocation


from supervisely.geometry.constants import FACES, POINTS, LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.point_location import points_to_row_col_list, row_col_list_to_points
from supervisely.geometry.rectangle import Rectangle


if not hasattr(np, 'bool'): np.bool = np.bool_

class CuboidFace:
    """
    CuboidFace for a single :class:`Cuboid<supervisely.geometry.cuboid.Cuboid>`.

    :param a: Node of the CuboidFace.
    :type a: int
    :param b: Node of the CuboidFace.
    :type b: int
    :param c: Node of the CuboidFace.
    :type c: int
    :param d: Node of the CuboidFace.
    :type d: int
    :Usage example:

     .. code-block:: python

        edge = CuboidFace(0, 1, 2, 3)
    """
    def __init__(self, a: int, b: int, c: int, d: int):
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def to_json(self) -> List[int]:
        """
        Convert the CuboidFace to list. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: List of integers
        :rtype: :class:`List[int, int, int, int]`
        :Usage example:

         .. code-block:: python

            edge = CuboidFace(0, 1, 2, 3)
            edge_json = edge.to_json()
            print(edge_json)
            # Output: [0, 1, 2, 3]
        """
        return [self.a, self.b, self.c, self.d]

    @classmethod
    def from_json(cls, data: List[int]) -> CuboidFace:
        """
        Convert list of integers to CuboidFace. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List of integers.
        :type data: List[int, int, int, int]
        :return: CuboidFace object
        :rtype: :class:`CuboidFace<CuboidFace>`
        :raises: :class:`ValueError` if data have not 4 indices
        :Usage example:

         .. code-block:: python

            new_edge = CuboidFace.from_json([0, 1, 2, 3])
        """
        if len(data) != 4:
            raise ValueError(f'CuboidFace JSON data must have exactly 4 indices, instead got {len(data)!r}.')
        return cls(data[0], data[1], data[2], data[3])

    @property
    def a(self):
        """
        """
        return self._a

    @property
    def b(self):
        """
        """
        return self._b

    @property
    def c(self):
        """
        """
        return self._c

    @property
    def d(self):
        """
        """
        return self._d

    def tolist(self) -> List[int]:
        """
        Convert CuboidFace to list.

        :return: List of integers.
        :rtype: :class:`List[int, int, int, int]`
        :Usage example:

         .. code-block:: python

            edge = CuboidFace(0, 1, 2, 3)
            print(edge.tolist())
            # Output: [0, 1, 2, 3]
        """
        return [self.a, self.b, self.c, self.d]


class Cuboid(Geometry):
    """
    Cuboid geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Cuboid<Cuboid>` class object is immutable.

    :param points: List or tuple of :class:`PointLocation<supervisely.geometry.point_location.PointLocation>` objects.
    :type points: List[PointLocation] or Tuple[PointLocation]
    :param faces: List or tuple of :class:`CuboidFace<CuboidFace>` objects.
    :type faces: List[CuboidFace] or Tuple[CuboidFace]
    :param sly_id: Cuboid ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID for :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which belongs Cuboid.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Cuboid.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Cuboid was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Cuboid was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`, if len(:class:`faces<faces>`) != 3

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.geometry.cuboid import CuboidFace

        nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
        edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
        pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
        figure = sly.Cuboid(pl_nodes, edges)
    """
    @staticmethod
    def geometry_name():
        """
        """
        return 'cuboid'

    def __init__(self, points: List[PointLocation], faces: List[CuboidFace], sly_id: Optional[int] = None, class_id: Optional[int] = None,
                 labeler_login: Optional[int] = None, updated_at: Optional[str] = None, created_at: Optional[str] = None):

        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

        points = list(points)
        faces = list(faces)

        if len(faces) != 3:
            raise ValueError(f'A cuboid must have exactly 3 faces. Instead got {len(faces)} faces.')

        for face in faces:
            for point_idx in (face.a, face.b, face.c, face.d):
                if point_idx >= len(points) or point_idx < 0:
                    raise ValueError(f'Point index is out of bounds for cuboid face. Got {len(points)!r} points, but '
                                     f'the index is {point_idx!r}.')

        self._points = points
        self._faces = faces

    @property
    def points(self) -> List[PointLocation]:
        """
        List of :class:`PointLocation<supervisely.geometry.point_location.PointLocation>` objects.

        :return: Cuboid nodes
        :rtype: :class:`List[PointLocation]`
        """
        return self._points.copy()

    @property
    def faces(self) -> List[CuboidFace]:
        """
        List of :class:`CuboidFace<CuboidFace>` objects.

        :return: Cuboid edges
        :rtype: :class:`List[CuboidFace]`
        """
        return self._faces.copy()

    def to_json(self) -> Dict:
        """
        Convert the Cuboid to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)

            figure_json = figure.to_json()
            print(figure_json)
            # Output: {
            #    "faces": [
            #        [0, 1, 2, 3],
            #        [0, 4, 5, 1],
            #        [1, 5, 6, 2]
            #    ],
            #    "points": [
            #        [273, 277],
            #        [273, 840],
            #        [690, 840],
            #        [690, 277],
            #        [168, 688],
            #        [168, 1200],
            #        [522, 1200]
            #    ]
            # }
        """
        packed_obj = {
            POINTS: points_to_row_col_list(self._points, flip_row_col_order=True),
            FACES: [face.to_json() for face in self._faces]
        }
        self._add_creation_info(packed_obj)
        return packed_obj

    @classmethod
    def from_json(cls, data: Dict) -> Cuboid:
        """
        Convert a json dict to Cuboid. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Cuboid in json format as a dict.
        :type data: dict
        :return: Cuboid object
        :rtype: :class:`Cuboid<Cuboid>`
        :raises: :class:`ValueError` if json format is not correct
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            data = {
                "faces": [
                            [0, 1, 2, 3],
                            [0, 4, 5, 1],
                            [1, 5, 6, 2]
                ],
                "points": [
                            [273, 277],
                            [273, 840],
                            [690, 840],
                            [690, 277],
                            [168, 688],
                            [168, 1200],
                            [522, 1200]
                ]
            }

            figure = sly.Cuboid.from_json(data)
        """
        for k in [POINTS, FACES]:
            if k not in data:
                raise ValueError(f'Field {k!r} not found in Cuboid JSON data.')

        points = row_col_list_to_points(data[POINTS], flip_row_col_order=True)
        faces = [CuboidFace.from_json(face_json) for face_json in data[FACES]]

        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(points=points, faces=faces,
                   sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    def crop(self, rect: Rectangle) -> List[Cuboid]:
        """
        Crops current Cuboid.

        :param rect: Rectangle object for crop.
        :type rect: Rectangle
        :return: List of Cuboid objects
        :rtype: :class:`List[Cuboid]<Cuboid>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)
            crop_figures = figure.crop(sly.Rectangle(1, 1, 1500, 1550))
        """
        is_all_nodes_inside = all(
            rect.contains_point_location(self._points[p]) for face in self._faces for p in face.tolist())
        return [self] if is_all_nodes_inside else []

    def _transform(self, transform_fn):
        """
        """
        return Cuboid(points=[transform_fn(p) for p in self.points], faces=self.faces)

    def rotate(self, rotator: ImageRotator) -> Cuboid:
        """
        Rotates current Cuboid.

        :param rotator: ImageRotator object for rotation.
        :type rotator: ImageRotator
        :return: Cuboid object
        :rtype: :class:`Cuboid<Cuboid>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace
            from supervisely.geometry.image_rotator import ImageRotator

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)

            # Remember that Cuboid class object is immutable, and we need to assign new instance of Cuboid to a new variable
            rotator = ImageRotator((300, 400), 25)
            rotate_fugure= figure.rotate(rotator)
        """
        return self._transform(lambda p: rotator.transform_point(p))

    def resize(self, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> Cuboid:
        """
        Resize current Cuboid.

        :param in_size: Input image size (height, width) to which belongs Cuboid object.
        :type in_size: Tuple[int, int]
        :param out_size: Desired output image size (height, width) to which belongs Cuboid object.
        :type out_size: Tuple[int, int]
        :return: Cuboid object
        :rtype: :class:`Cuboid<Cuboid>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)

            # Remember that Cuboid class object is immutable, and we need to assign new instance of Cuboid to a new variable
            in_height, in_width = 1250, 1400
            out_height, out_width = 600, 800
            resize_figure = figure.resize((in_height, in_width), (out_height, out_width))
        """
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor: float) -> Cuboid:
        """
        Scale current Cuboid.

        :param factor: Scale parameter.
        :type factor: float
        :return: Cuboid object
        :rtype: :class:`Cuboid<Cuboid>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)

            # Remember that Cuboid class object is immutable, and we need to assign new instance of Cuboid to a new variable
            scale_figure = figure.scale(1.75)
        """
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow: int, dcol: int) -> Cuboid:
        """
        Translate current Cuboid.

        :param drow: Horizontal shift.
        :type drow: int
        :param dcol: Vertical shift.
        :type dcol: int
        :return: Cuboid object
        :rtype: :class:`Cuboid<Cuboid>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)

            # Remember that Cuboid class object is immutable, and we need to assign new instance of Cuboid to a new variable
            translate_figure = figure.translate(150, 350)
        """
        return self._transform(lambda p: p.translate(drow, dcol))

    def fliplr(self, img_size: Tuple[int, int]) -> Cuboid:
        """
        Flips current Cuboid in horizontal.

        :param img_size: Image size (height, width) to which belongs Cuboid object.
        :type img_size: Tuple[int, int]
        :return: Cuboid object
        :rtype: :class:`Cuboid<Cuboid>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)

            # Remember that Cuboid class object is immutable, and we need to assign new instance of Cuboid to a new variable
            height, width = 1250, 1400
            fliplr_figure = figure.fliplr((height, width))
        """
        return self._transform(lambda p: p.fliplr(img_size))

    def flipud(self, img_size: Tuple[int, int]) -> Cuboid:
        """
        Flips current Cuboid in vertical.

        :param img_size: Image size (height, width) to which belongs Cuboid object.
        :type img_size: Tuple[int, int]
        :return: Cuboid object
        :rtype: :class:`Cuboid<Cuboid>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)

            # Remember that Cuboid class object is immutable, and we need to assign new instance of Cuboid to a new variable
            height, width = 1250, 1400
            flipud_figure = figure.flipud((height, width))
        """
        return self._transform(lambda p: p.flipud(img_size))

    def _draw_impl(self, bitmap: np.ndarray, color, thickness=1, config=None):
        """
        """
        bmp_to_draw = np.zeros(bitmap.shape[:2], np.uint8)
        for contour in self._contours_list():
            cv2.fillPoly(bmp_to_draw, pts=[np.array(contour, dtype=np.int32)], color=1)
        bool_mask = bmp_to_draw.astype(bool)
        bitmap[bool_mask] = color

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        """
        """
        contours_np_list = [np.array(contour, dtype=np.int32) for contour in self._contours_list()]
        cv2.polylines(bitmap, pts=contours_np_list, isClosed=True, color=color, thickness=thickness)

    def _contours_list(self):
        """
        """
        return [points_to_row_col_list([self._points[idx] for idx in face.tolist()], flip_row_col_order=True)
                for face in self._faces]

    def to_bbox(self) -> Rectangle:
        """
        Create Rectangle object from current Cuboid.

        :return: Rectangle object
        :rtype: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)

            rectangle = figure.to_bbox()
        """
        points_np = np.array([[self._points[p].row, self._points[p].col]
                              for face in self._faces for p in face.tolist()])
        rows, cols = points_np[:, 0], points_np[:, 1]
        return Rectangle(top=round(min(rows).item()), left=round(min(cols).item()), bottom=round(max(rows).item()),
                         right=round(max(cols).item()))

    @property
    def area(self) -> float:
        """
        Cuboid area.

        :return: Area of current Cuboid
        :rtype: :class:`float`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid import CuboidFace

            nodes = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
            edges = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
            pl_nodes = (sly.PointLocation(node[0], node[1]) for node in nodes)
            figure = sly.Cuboid(pl_nodes, edges)

            print(figure.area)
            # Output: 5146.0
        """
        bbox = self.to_bbox()
        canvas = np.zeros([bbox.bottom + 1, bbox.right + 1], dtype=np.bool)
        self.draw(canvas, True)
        return float(np.sum(canvas))
