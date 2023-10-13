# coding: utf-8
import base64
import zlib
import io
import numpy as np

from supervisely.geometry.bitmap_base import BitmapBase, resize_origin_and_bitmap
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.constants import MULTICHANNEL_BITMAP


class MultichannelBitmap(BitmapBase):
    """
    """
    @staticmethod
    def geometry_name():
        """
        """
        return 'multichannelBitmap'

    def __init__(self, data, origin: PointLocation = None,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        """
        The constructor for MultichannelBitmap class.
        :param data: bool numpy array
        :param origin: points (x and y coordinates) of the top left corner of a bitmap, i.e. the position of the
        bitmap within the image
        """
        super().__init__(data, origin, expected_data_dims=3,
                         sly_id=sly_id, class_id=class_id,
                         labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    @classmethod
    def _impl_json_class_name(cls):
        """
        """
        return MULTICHANNEL_BITMAP

    def rotate(self, rotator):
        """
        The function rotate render the bitmap within the full image canvas and rotate the whole canvas
        with a given rotator (ImageRotator class object contain size of image and angle to rotate)
        :param rotator: ImageRotator class object
        :return: MultichannelBitmap class object
        """
        full_img_data = np.zeros(rotator.src_imsize + self.data.shape[2:], dtype=self.data.dtype)
        full_img_data[
            self.origin.row:(self.origin.row + self.data.shape[0]),
            self.origin.col:(self.origin.col + self.data.shape[1]), ...] = self.data[:, :, ...]
        rotated_full_data = rotator.rotate_img(full_img_data, use_inter_nearest=True)
        # Rotate the bounding box to find out the bounding box of the rotated bitmap within the full image.
        rotated_bbox = self.to_bbox().rotate(rotator)
        rotated_origin = PointLocation(row=rotated_bbox.top, col=rotated_bbox.left)
        return MultichannelBitmap(data=rotated_bbox.get_cropped_numpy_slice(rotated_full_data), origin=rotated_origin)

    def crop(self, rect):
        """
        Crop the current MultichannelBitmap object with a given rectangle
        :param rect: Rectangle class object
        :return: MultichannelBitmap class object
        """
        maybe_cropped_area = self.to_bbox().crop(rect)
        if len(maybe_cropped_area) == 0:
            return []
        else:
            [cropped_area] = maybe_cropped_area
            cropped_origin = PointLocation(row=cropped_area.top, col=cropped_area.left)
            cropped_area_in_data = cropped_area.translate(drow=-self._origin.row, dcol=-self.origin.col)
            return [MultichannelBitmap(data=cropped_area_in_data.get_cropped_numpy_slice(self._data),
                                       origin=cropped_origin,)]

    def resize(self, in_size, out_size):
        """
        Resize the current MultichannelBitmap to match a certain size
        :param in_size: input image size
        :param out_size: output image size
        :return: MultichannelBitmap class object
        """
        scaled_origin, scaled_data = resize_origin_and_bitmap(self._origin, self._data, in_size, out_size)
        return MultichannelBitmap(data=scaled_data, origin=scaled_origin)

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """
        """
        self.to_bbox().get_cropped_numpy_slice(bitmap)[...] = color

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        """
        """
        bbox = self.to_bbox()
        # Not forwarding the config here directly since a Rectangle cannot know
        # about our config format.
        bbox.draw_contour(bitmap, color, thickness)

    @property
    def area(self):
        """
        :return: area of current MultichannelBitmap object
        """
        return self.data.shape[0] * self.data.shape[1]

    @staticmethod
    def base64_2_data(s: str) -> np.ndarray:
        """
        The function base64_2_data convert base64 encoded string to numpy
        :param s: string
        :return: numpy array
        """
        saved_bytes = io.BytesIO(zlib.decompress(base64.b64decode(s)))
        return np.load(saved_bytes)

    @staticmethod
    def data_2_base64(data: np.ndarray) -> str:
        """
        he function data_2_base64 convert numpy array to base64 encoded string
        :param data: numpy array
        :return: string
        """
        bytes_io = io.BytesIO()
        np.save(bytes_io, data, allow_pickle=False)
        return base64.b64encode(zlib.compress(bytes_io.getvalue())).decode('utf-8')

    def validate(self, name, settings):
        """
        """
        pass  # No need name validation - inner geometry type.
