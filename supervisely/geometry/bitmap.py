# coding: utf-8

# docs
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from supervisely.geometry.image_rotator import ImageRotator

import base64
from enum import Enum
import zlib
import io
import cv2
import numpy as np
from distutils.version import StrictVersion

from PIL import Image

from supervisely.geometry.bitmap_base import BitmapBase, resize_origin_and_bitmap
from supervisely.geometry.point_location import PointLocation, row_col_list_to_points
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.constants import BITMAP
from supervisely.imaging.image import read


if not hasattr(np, "bool"):
    np.bool = np.bool_


class SkeletonizeMethod(Enum):
    """
    Specifies possible skeletonization methods of :class:`Bitmap<Bitmap>`.
    """

    SKELETONIZE = 0
    """"""
    MEDIAL_AXIS = 1
    """"""
    THINNING = 2
    """"""


def _find_mask_tight_bbox(raw_mask: np.ndarray) -> Rectangle:
    rows = list(
        np.any(raw_mask, axis=1).tolist()
    )  # Redundant conversion to list to help PyCharm static analysis.
    cols = list(np.any(raw_mask, axis=0).tolist())
    top_margin = rows.index(True)
    bottom_margin = rows[::-1].index(True)
    left_margin = cols.index(True)
    right_margin = cols[::-1].index(True)
    return Rectangle(
        top=top_margin,
        left=left_margin,
        bottom=len(rows) - 1 - bottom_margin,
        right=len(cols) - 1 - right_margin,
    )


class Bitmap(BitmapBase):
    """
    Bitmap geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Bitmap<Bitmap>` object is immutable.

    :param data: Bitmap mask data. Must be a numpy array with only 2 unique values: [0, 1] or [0, 255] or [False, True].
    :type data: np.ndarray
    :param origin: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`: top, left corner of Bitmap. Position of the Bitmap within image.
    :type origin: PointLocation, optional
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
    :raises: :class:`ValueError`, if data is not bool or no pixels set to True in data
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create simple bitmap
        mask = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]], dtype=np.bool_)

        figure = sly.Bitmap(mask)

        # Note, when creating a bitmap, the specified mask is cut off by positive values, based on this, a origin is formed:
        print(figure.data)
        # Output:
        #    [[ True  True  True]
        #     [ True False  True]
        #     [ True  True  True]]

        origin = figure.origin.to_json()
        print(json.dumps(origin, indent=4))
        # Output: {
        #     "points": {
        #         "exterior": [
        #             [
        #                 1,
        #                 1
        #             ]
        #         ],
        #         "interior": []
        #     }

        # Create bitmap from black and white image:
        img = sly.imaging.image.read(os.path.join(os.getcwd(), 'black_white.jpeg'))
        mask = img[:, :, 0].astype(bool) # Get 2-dimensional bool numpy array
        figure = sly.Bitmap(mask)

     .. image:: https://i.imgur.com/2L3HRPs.jpg
    """

    @staticmethod
    def geometry_name():
        """geometry_name"""
        return "bitmap"

    def __init__(
        self,
        data: np.ndarray,
        origin: Optional[PointLocation] = None,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[int] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        if data.dtype != np.bool:
            unique, counts = np.unique(data, return_counts=True)
            if len(unique) != 2:
                raise ValueError(
                    f"Bitmap mask data must have only 2 unique values. Instead got {len(np.unique(data, return_counts=True)[0])}."
                )

            if list(unique) not in [[0, 1], [0, 255]]:
                raise ValueError(
                    f"Bitmap mask data values must be one of: [  0 1], [  0 255], [  False True]. Instead got {unique}."
                )

            if list(unique) == [0, 1]:
                data = np.array(data, dtype=bool)
            elif list(unique) == [0, 255]:
                data = np.array(data / 255, dtype=bool)

        # Call base constructor first to do the basic dimensionality checks.
        super().__init__(
            data=data,
            origin=origin,
            expected_data_dims=2,
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

        # Crop the empty margins of the boolean mask right away.
        if not np.any(data):
            raise ValueError(
                "Creating a bitmap with an empty mask (no pixels set to True) is not supported."
            )
        data_tight_bbox = _find_mask_tight_bbox(self._data)
        self._origin = self._origin.translate(drow=data_tight_bbox.top, dcol=data_tight_bbox.left)
        self._data = data_tight_bbox.get_cropped_numpy_slice(self._data)

    @classmethod
    def _impl_json_class_name(cls):
        """_impl_json_class_name"""
        return BITMAP

    def rotate(self, rotator: ImageRotator) -> Bitmap:
        """
        Rotates current Bitmap.

        :param rotator: :class:`ImageRotator<supervisely.geometry.image_rotator.ImageRotator>` for Bitamp rotation.
        :type rotator: ImageRotator
        :return: Bitmap object
        :rtype: :class:`Bitmap<Bitmap>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.image_rotator import ImageRotator

            height, width = ann.img_size
            rotator = ImageRotator((height, width), 25)
            # Remember that Bitmap class object is immutable, and we need to assign new instance of Bitmap to a new variable
            rotate_figure = figure.rotate(rotator)
        """
        full_img_mask = np.zeros(rotator.src_imsize, dtype=np.uint8)
        self.draw(full_img_mask, 1)
        # TODO this may break for one-pixel masks (it can disappear during rotation). Instead, rotate every pixel
        #  individually and set it in the resulting bitmap.
        new_mask = rotator.rotate_img(full_img_mask, use_inter_nearest=True).astype(np.bool)
        return Bitmap(data=new_mask)

    def crop(self, rect: Rectangle) -> List[Bitmap]:
        """
        Crops current Bitmap.

        :param rect: Rectangle object for cropping.
        :type rect: Rectangle
        :return: List of Bitmaps
        :rtype: :class:`List[Bitmap]<supervisely.geometry.bitmap.Bitmap>`

        :Usage Example:

         .. code-block:: python

            crop_figures = figure.crop(sly.Rectangle(1, 1, 300, 350))
        """
        maybe_cropped_bbox = self.to_bbox().crop(rect)
        if len(maybe_cropped_bbox) == 0:
            return []
        else:
            [cropped_bbox] = maybe_cropped_bbox
            cropped_bbox_relative = cropped_bbox.translate(
                drow=-self.origin.row, dcol=-self.origin.col
            )
            cropped_mask = cropped_bbox_relative.get_cropped_numpy_slice(self._data)
            if not np.any(cropped_mask):
                return []
            return [
                Bitmap(
                    data=cropped_mask,
                    origin=PointLocation(row=cropped_bbox.top, col=cropped_bbox.left),
                )
            ]

    def resize(self, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> Bitmap:
        """
        Resizes current Bitmap.

        :param in_size: Input image size (height, width) to which Bitmap belongs.
        :type in_size: Tuple[int, int]
        :param out_size: Output image size (height, width) to which Bitmap belongs.
        :type out_size: Tuple[int, int]
        :return: Bitmap object
        :rtype: :class:`Bitmap<Bitmap>`

        :Usage Example:

         .. code-block:: python

            in_height, in_width = 800, 1067
            out_height, out_width = 600, 800
            # Remember that Bitmap class object is immutable, and we need to assign new instance of Bitmap to a new variable
            resize_figure = figure.resize((in_height, in_width), (out_height, out_width))
        """
        scaled_origin, scaled_data = resize_origin_and_bitmap(
            self._origin, self._data.astype(np.uint8), in_size, out_size
        )
        # TODO this might break if a sparse mask is resized too thinly. Instead, resize every pixel individually and set
        #  it in the resulting bitmap.
        return Bitmap(data=scaled_data.astype(np.bool), origin=scaled_origin)

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """_draw_impl"""
        self.to_bbox().get_cropped_numpy_slice(bitmap)[self.data] = color

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        """_draw_contour_impl"""
        if StrictVersion(cv2.__version__) >= StrictVersion("4.0.0"):
            contours, _ = cv2.findContours(
                self.data.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            _, contours, _ = cv2.findContours(
                self.data.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
        if contours is not None:
            for cont in contours:
                cont[:, :] += (
                    self.origin.col,
                    self.origin.row,
                )  # cont with shape (rows, ?, 2)
            cv2.drawContours(bitmap, contours, -1, color, thickness=thickness)

    @property
    def area(self) -> float:
        """
        Bitmap area.

        :return: Area of current Bitmap
        :rtype: :class:`float`
        :Usage example:

         .. code-block:: python

            print(figure.area)
            # Output: 88101.0
        """
        return float(self._data.sum())

    @staticmethod
    def base64_2_data(s: str) -> np.ndarray:
        """
        Convert base64 encoded string to numpy array.

        :param s: Input base64 encoded string.
        :type s: str
        :return: Bool numpy array
        :rtype: :class:`np.ndarray`
        :Usage example:

         .. code-block:: python

              import supervisely as sly

              encoded_string = 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'
              figure_data = sly.Bitmap.base64_2_data(encoded_string)
              print(figure_data)
              #  [[ True  True  True]
              #   [ True False  True]
              #   [ True  True  True]]
        """
        z = zlib.decompress(base64.b64decode(s))
        n = np.frombuffer(z, np.uint8)

        imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
        if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] >= 4):
            mask = imdecoded[:, :, 3].astype(bool)  # 4-channel imgs
        elif len(imdecoded.shape) == 2:
            mask = imdecoded.astype(bool)  # flat 2d mask
        else:
            raise RuntimeError("Wrong internal mask format.")
        return mask

    @staticmethod
    def data_2_base64(mask: np.ndarray) -> str:
        """
        Convert numpy array to base64 encoded string.

        :param mask: Bool numpy array.
        :type mask: np.ndarray
        :return: Base64 encoded string
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get annotation from API
            meta_json = api.project.get_meta(PROJECT_ID)
            meta = sly.ProjectMeta.from_json(meta_json)
            ann_info = api.annotation.download(IMAGE_ID)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            # Get Bitmap from annotation
            for label in ann.labels:
                if type(label.geometry) == sly.Bitmap:
                    figure = label.geometry

            encoded_string = sly.Bitmap.data_2_base64(figure.data)
            print(encoded_string)
            # 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'
        """
        img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
        img_pil.putpalette([0, 0, 0, 255, 255, 255])
        bytes_io = io.BytesIO()
        img_pil.save(bytes_io, format="PNG", transparency=0, optimize=0)
        bytes_enc = bytes_io.getvalue()
        return base64.b64encode(zlib.compress(bytes_enc)).decode("utf-8")

    def skeletonize(self, method_id: SkeletonizeMethod) -> Bitmap:
        """
        Compute the skeleton, medial axis transform or morphological thinning of Bitmap.

        :param method_id: Method to convert bool numpy array.
        :type method_id: SkeletonizeMethod
        :return: Bitmap object
        :rtype: :class:`Bitmap<Bitmap>`
        :Usage example:

         .. code-block:: python

            # Remember that Bitmap class object is immutable, and we need to assign new instance of Bitmap to a new variable
            skeleton_figure = figure.skeletonize(SkeletonizeMethod.SKELETONIZE)
            med_ax_figure = figure.skeletonize(SkeletonizeMethod.MEDIAL_AXIS)
            thin_figure = figure.skeletonize(SkeletonizeMethod.THINNING)
        """
        from skimage import morphology as skimage_morphology

        if method_id == SkeletonizeMethod.SKELETONIZE:
            method = skimage_morphology.skeletonize
        elif method_id == SkeletonizeMethod.MEDIAL_AXIS:
            method = skimage_morphology.medial_axis
        elif method_id == SkeletonizeMethod.THINNING:
            method = skimage_morphology.thin
        else:
            raise NotImplementedError("Method {!r} does't exist.".format(method_id))

        mask_u8 = self.data.astype(np.uint8)
        res_mask = method(mask_u8).astype(bool)
        return Bitmap(data=res_mask, origin=self.origin)

    def to_contours(self) -> List[Polygon]:
        """
        Get list of contours in Bitmap.

        :return: List of Polygon objects
        :rtype: :class:`List[Polygon]<supervisely.geometry.polygon.Polygon>`
        :Usage example:

         .. code-block:: python

            figure_contours = figure.to_contours()
        """
        origin, mask = self.origin, self.data
        if StrictVersion(cv2.__version__) >= StrictVersion("4.0.0"):
            contours, hier = cv2.findContours(
                mask.astype(np.uint8),
                mode=cv2.RETR_CCOMP,  # two-level hierarchy, to get polygons with holes
                method=cv2.CHAIN_APPROX_SIMPLE,
            )
        else:
            _, contours, hier = cv2.findContours(
                mask.astype(np.uint8),
                mode=cv2.RETR_CCOMP,  # two-level hierarchy, to get polygons with holes
                method=cv2.CHAIN_APPROX_SIMPLE,
            )
        if (hier is None) or (contours is None):
            return []

        res = []
        for idx, hier_pos in enumerate(hier[0]):
            next_idx, prev_idx, child_idx, parent_idx = hier_pos
            if parent_idx < 0:
                external = contours[idx][:, 0].tolist()
                internals = []
                while child_idx >= 0:
                    internals.append(contours[child_idx][:, 0])
                    child_idx = hier[0][child_idx][0]
                if len(external) > 2:
                    new_poly = Polygon(
                        exterior=row_col_list_to_points(external, flip_row_col_order=True),
                        interior=[
                            row_col_list_to_points(x, flip_row_col_order=True) for x in internals
                        ],
                    )
                    res.append(new_poly)

        offset_row, offset_col = origin.row, origin.col
        res = [x.translate(offset_row, offset_col) for x in res]
        return res

    def bitwise_mask(self, full_target_mask: np.ndarray, bit_op) -> Bitmap:
        """
        Make bitwise operations between a given numpy array and Bitmap.

        :param full_target_mask: Input numpy array.
        :type full_target_mask: np.ndarray
        :param bit_op: Type of bitwise operation(and, or, not, xor), uses `numpy logic <https://numpy.org/doc/stable/reference/routines.logic.html>`_ functions.
        :type bit_op: `Numpy logical operation <https://numpy.org/doc/stable/reference/routines.logic.html#logical-operations>`_
        :return: Bitmap object or empty list
        :rtype: :class:`Bitmap<Bitmap>` or :class:`list`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            mask = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 0, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]], dtype=np.bool_)

            figure = sly.Bitmap(mask)

            array = np.array([[0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=np.bool_)

            bitwise_figure = figure.bitwise_mask(array, np.logical_and)
            print(bitwise_figure.data)
            # Output:
            # [[ True  True  True]
            #  [False False False]
            #  [False  True False]]
        """
        full_size = full_target_mask.shape[:2]
        origin, mask = self.origin, self.data
        full_size_mask = np.full(full_size, False, bool)
        full_size_mask[
            origin.row : origin.row + mask.shape[0],
            origin.col : origin.col + mask.shape[1],
        ] = mask

        new_mask = bit_op(full_target_mask, full_size_mask).astype(bool)
        if new_mask.sum() == 0:
            return []
        new_mask = new_mask[
            origin.row : origin.row + mask.shape[0],
            origin.col : origin.col + mask.shape[1],
        ]
        return Bitmap(data=new_mask, origin=origin.clone())

    @classmethod
    def allowed_transforms(cls):
        """allowed_transforms"""
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.polygon import Polygon
        from supervisely.geometry.rectangle import Rectangle

        return [AnyGeometry, Polygon, Rectangle]

    @classmethod
    def from_path(cls, path: str) -> Bitmap:
        """
        Read bitmap from image by path.

        :param path: Path to image
        :type path: str
        :return: Bitmap
        :rtype: Bitmap
        """
        img = read(path)
        return Bitmap(img[:, :, 0])
