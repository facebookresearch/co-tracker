# coding: utf-8

import math

import cv2
import numpy as np
from typing import List, Tuple

from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.point_location import PointLocation


# to rotate image & objects wrt source img center
# output image will contain all 'pixels' from source img
class ImageRotator:
    """
    ImageRotator for rotating images and geometry figures.

    :param imsize: Shape of the image (height, width)
    :type imsize: Tuple[int, int] or List[int, int]
    :param angle_degrees_ccw: Angle to rotate image.
    :type angle_degrees_ccw: int

    :Usage example:

     .. code-block:: python

        height, width = 300, 400
        rotator = ImageRotator((height, width), 25)
    """
    # to get rect with max 'coloured' area in rotated img
    def _calc_inner_crop(self):
        """
        Given a rectangle of self.src_imsize HxW that has been rotated by
        self.angle_degrees_ccw (in degrees), computes the location of the
        largest possible axis-aligned rectangle within the rotated rectangle.
        """

        # TODO This needs significant streamlinig.
        a_ccw = np.deg2rad(self.angle_degrees_ccw)
        quadrant = math.floor(a_ccw / (math.pi / 2)) & 3
        sign_alpha = a_ccw if ((quadrant & 1) == 0) else math.pi - a_ccw
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        h, w = self.src_imsize
        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        largest_w, largest_h = bb_w - 2 * x, bb_h - 2 * y

        new_h, new_w = self.new_imsize
        left = round((new_w - largest_w) * 0.5)
        right = round((new_w + largest_w) * 0.5)
        top = round((new_h - largest_h) * 0.5)
        bottom = round((new_h + largest_h) * 0.5)
        some_inner_crop = Rectangle(top, left, bottom, right)
        new_img_bbox = Rectangle(0, 0, self.new_imsize[0] - 1, self.new_imsize[1] - 1)
        self.inner_crop = new_img_bbox.crop(some_inner_crop)[0]

    @staticmethod
    def _affine_matrix_and_new_canvas_size(imsize, angle_degrees_ccw):
        """
        """
        rows, cols = imsize
        bottom, right = rows - 1, cols - 1
        image_center = (bottom / 2.0, right / 2.0)
        # Opencv uses a left-hand-side coordinate system (col, row), so to get the same effective transform we need to
        # negate the angle.
        affine_matrix = cv2.getRotationMatrix2D(image_center, -angle_degrees_ccw, 1.0)
        source_corners_uniform = np.array([
            [0, 0, 1],
            [0, right, 1],
            [bottom, 0, 1],
            [bottom, right, 1]])
        rotated_corners = affine_matrix.dot(source_corners_uniform.T).T

        # Get the upper and lower calues for rotated coordinates to find out the bounding box.
        rotated_upper_values = np.max(rotated_corners, axis=0)
        rotated_lower_values = np.min(rotated_corners, axis=0)

        # Add extra translation to map the upper left corner of the boundig box to origin (i.e. to get coordinates in
        # the new canvas covering the entire rotated image.
        affine_matrix[:, 2] -= rotated_lower_values
        new_canvas_size = np.ceil(rotated_upper_values - rotated_lower_values + 1).astype(np.int64).tolist()
        return affine_matrix, new_canvas_size

    def __init__(self, imsize: Tuple[int, int], angle_degrees_ccw: int):
        self.src_imsize = tuple(imsize)
        self.angle_degrees_ccw = angle_degrees_ccw
        # Transform matrix for the RHS (col, row) coordinate system.
        self.affine_matrix, self.new_imsize = self._affine_matrix_and_new_canvas_size(imsize, angle_degrees_ccw)
        # Opencv uses a left-hand-side coordinate system (col, row), so to get the same effective transform we need to
        # negate the angle.
        # Also flip the image size as OpenCV uses cols-rows order.
        self.opencv_affine_matrix, _ = self._affine_matrix_and_new_canvas_size(imsize[::-1], -angle_degrees_ccw)
        self._calc_inner_crop()

    def transform_point(self, point: PointLocation) -> PointLocation:
        """
        Calculates new parameters of PointLocation after rotation.

        :param point: PointLocation object.
        :type point: PointLocation
        :return: PointLocation object
        :rtype: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            point = sly.PointLocation(100, 200)
            height, width = 300, 400
            rotator = ImageRotator((height, width), 25)

            rotate_point = rotator.transform_point(point)
            rotate_point_json = rotate_point.to_json()
            print(rotate_point_json)
            # Output:
            # {
            #    "points": {
            #        "exterior": [
            #            [224, 175]
            #        ],
            #        "interior": []
            #    }
            # }
        """
        point_np_uniform = np.array([point.row, point.col, 1])
        transformed_np = self.affine_matrix.dot(point_np_uniform)
        # Unwrap numpy types so that round() produces integer results.
        return PointLocation(row=round(transformed_np[0].item()), col=round(transformed_np[1].item()))

    def rotate_img(self, img: np.ndarray, use_inter_nearest: bool) -> np.ndarray:
        """
        Calculates new parameters of image after rotation.

        :param img: Image to rotate.
        :type img: np.ndarray
        :param use_inter_nearest: If True uses `cv2.INTER_NEAREST <https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121aa5521d8e080972c762467c45f3b70e6c>`_ parameter in rotation, otherwise don't.
        :type use_inter_nearest: bool
        :return: Rotated image
        :rtype: :class:`np.ndarray`
        :Usage example:

         .. code-block:: python

            height, width = 300, 400
            rotator = ImageRotator((height, width), 25)
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            rotate_mask = rotator.rotate_img(mask, True)
            print(rotate_mask.shape)
            # Output: (441, 489, 3)
        """
        if use_inter_nearest:
            interp = cv2.INTER_NEAREST  # @TODO: cv2 INTER_NEAREST may shift coords, what to do?
        else:
            interp = cv2.INTER_LANCZOS4
        res = cv2.warpAffine(src=img, M=self.opencv_affine_matrix, dsize=tuple(self.new_imsize[::-1]), flags=interp)
        return res

