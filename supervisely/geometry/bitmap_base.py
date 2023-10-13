# coding: utf-8
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional

from supervisely.geometry.constants import DATA, ORIGIN, GEOMETRY_SHAPE, GEOMETRY_TYPE, \
                                               LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging.image import resize_inter_nearest, restore_proportional_size


if not hasattr(np, 'bool'): np.bool = np.bool_

# TODO: rename to resize_bitmap_and_origin
def resize_origin_and_bitmap(origin: PointLocation, bitmap: np.ndarray, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> Tuple[PointLocation, np.ndarray]:
    """
    Change PointLocation and resize numpy array to match a certain size.

    :param origin: PointLocation to resize.
    :type origin: PointLocation
    :param bitmap: Numpy array to resize.
    :type bitmap: np.ndarray
    :param in_size: Input image size (height, width) to which belongs :class:`PointLocation<supervisely.geometry.point_location.PointLocation>` object and numpy array.
    :type in_size: Tuple[int, int]
    :param out_size: Desired output image size (height, width) to which belongs :class:`PointLocation<supervisely.geometry.point_location.PointLocation>` object and numpy array.
    :type out_size: Tuple[int, int]

    :return: PointLocation object and numpy array
    :rtype: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`, :class:`np.ndarray`
    :Usage Example:

     .. code-block:: python

        resize_origin, resize_bitmap = resize_origin_and_bitmap(origin, bitmap, (400, 500), (800, 1000))
    """
    new_size = restore_proportional_size(in_size=in_size, out_size=out_size)

    row_scale = new_size[0] / in_size[0]
    col_scale = new_size[1] / in_size[1]

    # TODO: Double check (+restore_proportional_size) or not? bitmap.shape and in_size are equal?
    # Make sure the resulting size has at least one pixel in every direction (i.e. limit the shrinkage to avoid having
    # empty bitmaps as a result).
    scaled_rows = max(round(bitmap.shape[0] * row_scale), 1)
    scaled_cols = max(round(bitmap.shape[1] * col_scale), 1)

    scaled_origin = PointLocation(row=round(origin.row * row_scale), col=round(origin.col * col_scale))
    scaled_bitmap = resize_inter_nearest(bitmap, (scaled_rows, scaled_cols))
    return scaled_origin, scaled_bitmap


class BitmapBase(Geometry):
    """
    BitmapBase is a base class of :class:`Bitmap<supervisely.geometry.bitmap.Bitmap>` geometry. :class:`BitmapBase<BitmapBase>` class object is immutable.

    :param data: Bitmap mask data.
    :type data: np.ndarray
    :param origin: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`: top, left corner of Bitmap. Position of the Bitmap within image.
    :type origin: PointLocation, optional
    :param expected_data_dims: Number of dimensions of data numpy array.
    :type expected_data_dims: int, optional
    :param sly_id: Bitmap ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Bitmap belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Bitmap.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Bitmap was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Bitmap was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional

    :Usage example: Example of creating and using see in :class:`Bitmap<supervisely.geometry.bitmap.Bitmap>`.
    """
    def __init__(self, data: np.ndarray, origin: Optional[PointLocation] = None, expected_data_dims: Optional[int] = None,
                 sly_id: Optional[int] = None, class_id: Optional[int] = None, labeler_login: Optional[int] = None,
                 updated_at: Optional[str] = None, created_at: Optional[str] = None):
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
        if origin is None:
            origin = PointLocation(row=0, col=0)

        if not isinstance(origin, PointLocation):
            raise TypeError('BitmapBase "origin" argument must be "PointLocation" object!')

        if not isinstance(data, np.ndarray):
            raise TypeError('BitmapBase "data" argument must be numpy array object!')

        data_dims = len(data.shape)
        if expected_data_dims is not None and data_dims != expected_data_dims:
            raise ValueError(f'BitmapBase "data" argument must be a {expected_data_dims}-dimensional numpy array. Instead got {data_dims} dimensions')


        self._origin = origin.clone()
        self._data = data.copy()

    @classmethod
    def _impl_json_class_name(cls):
        """Descendants must implement this to return key string to look up serialized representation in a JSON dict."""
        raise NotImplementedError()

    @staticmethod
    def base64_2_data(s: str) -> np.ndarray:
        """
        """
        raise NotImplementedError()

    @staticmethod
    def data_2_base64(data: np.ndarray) -> str:
        """
        """
        raise NotImplementedError()

    def to_json(self) -> Dict:
        """
        Convert the BitmapBase to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            mask = np.array([[0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 1, 0, 1, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0]], dtype=np.bool_)

            figure = sly.Bitmap(mask)
            figure_json = figure.to_json()
            print(json.dumps(figure_json, indent=4))
            # Output: {
            #    "bitmap": {
            #        "origin": [1, 1],
            #        "data": "eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6"
            #    },
            #    "shape": "bitmap",
            #    "geometryType": "bitmap"
            # }
        """
        res = {
            self._impl_json_class_name(): {
                ORIGIN: [self.origin.col, self.origin.row],
                DATA: self.data_2_base64(self.data)
            },
            GEOMETRY_SHAPE: self.geometry_name(),
            GEOMETRY_TYPE: self.geometry_name(),
        }
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, json_data: Dict) -> BitmapBase:
        """
        Convert a json dict to BitmapBase. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Bitmap in json format as a dict.
        :type data: dict
        :return: BitmapBase object
        :rtype: :class:`BitmapBase<BitmapBase>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_json = {
                "bitmap": {
                    "origin": [1, 1],
                    "data": "eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6"
                },
                "shape": "bitmap",
                "geometryType": "bitmap"
            }

            figure = sly.Bitmap.from_json(figure_json)
        """
        json_root_key = cls._impl_json_class_name()
        if json_root_key not in json_data:
            raise ValueError(
                'Data must contain {} field to create MultichannelBitmap object.'.format(json_root_key))

        if ORIGIN not in json_data[json_root_key] or DATA not in json_data[json_root_key]:
            raise ValueError('{} field must contain {} and {} fields to create MultichannelBitmap object.'.format(
                json_root_key, ORIGIN, DATA))

        col, row = json_data[json_root_key][ORIGIN]
        data = cls.base64_2_data(json_data[json_root_key][DATA])

        labeler_login = json_data.get(LABELER_LOGIN, None)
        updated_at = json_data.get(UPDATED_AT, None)
        created_at = json_data.get(CREATED_AT, None)
        sly_id = json_data.get(ID, None)
        class_id = json_data.get(CLASS_ID, None)
        return cls(data=data, origin=PointLocation(row=row, col=col),
                   sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    @property
    def origin(self) -> PointLocation:
        """
        Position of the Bitmap within image.

        :return: Top, left corner of Bitmap.
        :rtype: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`
        """
        return self._origin.clone()

    @property
    def data(self) -> np.ndarray:
        """
        Get mask data of Bitmap.

        :return: Data of Bitmap.
        :rtype: :class:`np.ndarray`
        """
        return self._data.copy()

    def translate(self, drow: int, dcol: int) -> BitmapBase:
        """
        Translate current Bitmap.

        :param drow: Horizontal shift.
        :type drow: int
        :param dcol: Vertical shift.
        :type dcol: int
        :return: BitmapBase object
        :rtype: :class:`BitmapBase<BitmapBase>`

        :Usage Example:

         .. code-block:: python

            # Remember that Bitmap class object is immutable, and we need to assign new instance of Bitmap to a new variable
            translate_figure = figure.translate(150, 250)
        """
        translated_origin = self.origin.translate(drow, dcol)
        return self.__class__(data=self.data, origin=translated_origin)

    def fliplr(self, img_size: Tuple[int, int]) -> BitmapBase:
        """
        Flip current Bitmap in horizontal.

        :param img_size: :class:`Annotation.img_size<supervisely.annotation.annotation.Annotation.img_size>` which belongs Bitmap.
        :type img_size: Tuple[int, int]
        :return: BitmapBase object
        :rtype: :class:`BitmapBase<BitmapBase>`

        :Usage Example:

         .. code-block:: python

            # Remember that Bitmap class object is immutable, and we need to assign new instance of Bitmap to a new variable
            height, width = 300, 400
            fliplr_figure = figure.fliplr((height, width))
        """
        flipped_mask = np.flip(self.data, axis=1)
        flipped_origin = PointLocation(row=self.origin.row, col=(img_size[1] - flipped_mask.shape[1] - self.origin.col))
        return self.__class__(data=flipped_mask, origin=flipped_origin)

    def flipud(self, img_size: Tuple[int, int]) -> BitmapBase:
        """
        Flip current Bitmap in vertical.

        :param img_size: :class:`Annotation.img_size<supervisely.annotation.annotation.Annotation.img_size>` which belongs Bitmap.
        :type img_size: Tuple[int, int]
        :return: BitmapBase object
        :rtype: :class:`BitmapBase<BitmapBase>`

        :Usage Example:

         .. code-block:: python

            # Remember that Bitmap class object is immutable, and we need to assign new instance of Bitmap to a new variable
            height, width = 300, 400
            flipud_figure = figure.flipud((height, width))
        """
        flipped_mask = np.flip(self.data, axis=0)
        flipped_origin = PointLocation(row=(img_size[0] - flipped_mask.shape[0] - self.origin.row), col=self.origin.col)
        return self.__class__(data=flipped_mask, origin=flipped_origin)

    def scale(self, factor: float) -> BitmapBase:
        """
        Scale current Bitmap.

        :param factor: Scale parameter.
        :type factor: float
        :return: BitmapBase object
        :rtype: :class:`BitmapBase<BitmapBase>`

        :Usage Example:

         .. code-block:: python

            # Remember that Bitmap class object is immutable, and we need to assign new instance of Bitmap to a new variable
            scale_figure = figure.scale(0.75)
        """
        new_rows = round(self._data.shape[0] * factor)
        new_cols = round(self._data.shape[1] * factor)
        mask = self._resize_mask(self.data, new_rows, new_cols)
        origin = self.origin.scale(factor)
        return self.__class__(data=mask, origin=origin)

    @staticmethod
    def _resize_mask(mask, out_rows, out_cols):
        """
        """
        return resize_inter_nearest(mask.astype(np.uint8), (out_rows, out_cols)).astype(np.bool)

    def to_bbox(self) -> Rectangle:
        """
        Create :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>` object from current Bitmap.

        :return: Rectangle object
        :rtype: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`

        :Usage Example:

         .. code-block:: python

            rectangle = figure.to_bbox()
        """
        return Rectangle.from_array(self._data).translate(drow=self._origin.row, dcol=self._origin.col)
