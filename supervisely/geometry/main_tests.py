# coding: utf-8
import unittest
import numpy as np
import json

from supervisely.geometry.point_location import PointLocation, row_col_list_to_points, points_to_row_col_list
from supervisely.geometry.point import Point
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.imaging.image import KEEP_ASPECT_RATIO
from supervisely.geometry.constants import POINTS, EXTERIOR, INTERIOR, BITMAP, DATA, ORIGIN


if not hasattr(np, 'bool'): np.bool = np.bool_

class PointTest(unittest.TestCase):
    def setUp(self):
        self.point = Point(row=10,  col=5)

    def assertPointEquals(self, point, row, col):
        self.assertIsInstance(point, Point)
        self.assertEqual(point.row, row)
        self.assertEqual(point.col, col)

    def test_empty_crop(self):
        crop_rect = Rectangle(100, 100, 200, 200)
        res_geoms = self.point.crop(crop_rect)
        self.assertEqual(len(res_geoms), 0)

    def test_crop(self):
        crop_rect = Rectangle(0, 0, 100, 100)
        res_geoms = self.point.crop(crop_rect)
        self.assertEqual(len(res_geoms), 1)
        res_point = res_geoms[0]
        self.assertPointEquals(res_point, self.point.row, self.point.col)

    def test_relative_crop(self):
        crop_rect = Rectangle(5, 5, 100, 100)
        res_geoms = self.point.relative_crop(crop_rect)
        self.assertEqual(len(res_geoms), 1)
        res_point = res_geoms[0]
        self.assertPointEquals(res_point, 5, 0)

    def test_rotate(self):
        rotator = ImageRotator((21, 21), 90)  # Center pixel is (10, 10)
        rot_point = self.point.rotate(rotator)
        self.assertPointEquals(rot_point, 15, 10)

    def test_resize(self):
        in_size = (20, 40)
        out_size = (30, KEEP_ASPECT_RATIO)
        res_point = self.point.resize(in_size, out_size)
        self.assertPointEquals(res_point, 15, 8)

    def test_scale(self):
        factor = 2.7
        res_point = self.point.scale(factor)
        self.assertPointEquals(res_point, 27, 14)

    def test_translate(self):
        drow = 21
        dcol = -9
        res_point = self.point.translate(drow=drow, dcol=dcol)
        self.assertPointEquals(res_point, 31, -4)

    def test_fliplr(self):
        imsize = (110, 100)
        res_point = self.point.fliplr(imsize)
        self.assertPointEquals(res_point, self.point.row, 95)

    def test_flipud(self):
        imsize = (90, 100)
        res_point = self.point.flipud(imsize)
        self.assertPointEquals(res_point, 80, self.point.col)

    def test_area(self):
        area = self.point.area
        self.assertIsInstance(area, float)
        self.assertEqual(area, 0.0)

    def test_to_bbox(self):
        rect = self.point.to_bbox()
        self.assertPointEquals(self.point, rect.top, rect.left)
        self.assertPointEquals(self.point, rect.bottom, rect.right)

    def test_clone(self):
        res_point = self.point.clone()
        self.assertPointEquals(res_point, self.point.row, self.point.col)
        self.assertIsNot(res_point, self.point)

    def test_from_json(self):
        packed_obj = {
            'some_stuff': 'aaa',
            POINTS: {
                EXTERIOR: [[17, 3]],
                INTERIOR: []
            }
        }
        res_point = Point.from_json(packed_obj)
        self.assertIsInstance(res_point, Point)
        self.assertEqual(res_point.row, 3)
        self.assertEqual(res_point.col, 17)

    def test_to_json(self):
        res_obj = self.point.to_json()
        expected_dict = {
            POINTS: {
                EXTERIOR: [[5, 10]],
                INTERIOR: []
            }
        }
        self.assertDictEqual(res_obj, expected_dict)


class RectangleTest(unittest.TestCase):
    def setUp(self):
        self.rect = Rectangle(top=5, left=10, bottom=30, right=30)

    def assertRectEquals(self, rect, top, left, bottom, right):
        self.assertIsInstance(rect, Rectangle)
        self.assertEqual(rect.top, top)
        self.assertEqual(rect.left, left)
        self.assertEqual(rect.bottom, bottom)
        self.assertEqual(rect.right, right)

    def test_empty_crop(self):
        crop_rect = Rectangle(100, 100, 200, 200)
        res_geoms = self.rect.crop(crop_rect)
        self.assertEqual(len(res_geoms), 0)

    def test_crop(self):
        crop_rect = Rectangle(0, 0, 100, 100)
        res_geoms = self.rect.crop(crop_rect)
        self.assertEqual(len(res_geoms), 1)
        res_rect = res_geoms[0]
        self.assertRectEquals(res_rect, self.rect.top, self.rect.left, self.rect.bottom, self.rect.right)

    def test_relative_crop(self):
        crop_rect = Rectangle(3, 4, 100, 100)
        res_geoms = self.rect.relative_crop(crop_rect)
        self.assertEqual(len(res_geoms), 1)
        res_rect = res_geoms[0]
        self.assertRectEquals(res_rect, 2, 6, 27, 26)

    def test_rotate(self):
        imsize = (101, 101)
        rotator = ImageRotator(imsize=imsize, angle_degrees_ccw=90)
        res_rect = self.rect.rotate(rotator)
        self.assertRectEquals(res_rect, 70, 5, 90, 30)

    def test_resize(self):
        in_size = (100, 100)
        out_size = (200, 150)
        res_rect = self.rect.resize(in_size, out_size)
        self.assertRectEquals(res_rect, 10, 15, 60, 45)

    def test_scale(self):
        factor = 1.3
        res_rect = self.rect.scale(factor)
        self.assertRectEquals(res_rect, 6, 13, 39, 39)

    def test_translate(self):
        drows = 8
        dcols = 259
        res_rect = self.rect.translate(drows, dcols)
        self.assertRectEquals(res_rect, 13, 269, 38, 289)

    def test_fliplr(self):
        im_size = (100, 200)
        res_rect = self.rect.fliplr(im_size)
        self.assertRectEquals(res_rect, 5, 170, 30, 190)

    def test_flipud(self):
        im_size = (100, 200)
        res_rect = self.rect.flipud(im_size)
        self.assertRectEquals(res_rect, 70, 10, 95, 30)

    def test_area(self):
        area = self.rect.area
        self.assertIsInstance(area, float)
        self.assertEqual(area, 546.0)

    def test_to_bbox(self):
        res_rect = self.rect.to_bbox()
        self.assertRectEquals(res_rect, self.rect.top, self.rect.left, self.rect.bottom, self.rect.right)

    def test_clone(self):
        res_rect = self.rect.clone()
        self.assertRectEquals(res_rect, self.rect.top, self.rect.left, self.rect.bottom, self.rect.right)
        self.assertIsNot(res_rect, self.rect)

    def test_from_json(self):
        packed_obj = {
            'some_stuff': 'aaa',
            POINTS: {
                EXTERIOR: [[17, 3], [34, 45]],
                INTERIOR: []
            }
        }
        res_rect = Rectangle.from_json(packed_obj)
        self.assertRectEquals(res_rect, 3, 17, 45, 34)

    def test_to_json(self):
        res_obj = self.rect.to_json()
        expected_dict = {
            POINTS: {
                EXTERIOR: [[10, 5], [30, 30]],
                INTERIOR: []
            }
        }
        self.assertDictEqual(res_obj, expected_dict)

    def test_draw(self):
        rect = Rectangle(1, 1, 3, 3)

        bitmap_1 = np.zeros((5, 5), dtype=np.uint8)
        rect.draw_contour(bitmap_1, 1)
        expected_mask_1 = np.array([[0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(bitmap_1, expected_mask_1))

        bitmap_2 = np.zeros((5, 5), dtype=np.uint8)
        rect.draw(bitmap_2, 1)
        expected_mask_2 = np.array([[0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(bitmap_2, expected_mask_2))


class PolygonTest(unittest.TestCase):
    def setUp(self):
        self.exterior = [[10, 10], [40, 10], [30, 40], [10, 30]]
        self.interiors = [[[20, 20], [30, 20], [30, 30], [20, 30]]]
        self.poly = Polygon(exterior=row_col_list_to_points(self.exterior, flip_row_col_order=True),
                            interior=[row_col_list_to_points(self.interiors[0], flip_row_col_order=True)])

    def assertPolyEquals(self, poly, exterior, interiors):
        self.assertIsInstance(poly, Polygon)
        self.assertCountEqual(points_to_row_col_list(poly.exterior, flip_row_col_order=True), exterior)
        self.assertEqual(len(poly.interior), len(interiors))
        for p_interior, interior in zip(poly.interior, interiors):
            self.assertCountEqual(points_to_row_col_list(p_interior, flip_row_col_order=True), interior)

    def test_empty_crop(self):
        crop_rect = Rectangle(100, 100, 200, 200)
        res_geoms = self.poly.crop(crop_rect)
        self.assertEqual(len(res_geoms), 0)

    def test_crop(self):
        crop_rect = Rectangle(25, 0, 200, 200)
        res_geoms = self.poly.crop(crop_rect)
        self.assertEqual(len(res_geoms), 1)
        crop = res_geoms[0]
        self.assertPolyEquals(crop,
                              [[10, 25], [20, 25], [20, 30], [30, 30], [30, 25], [35, 25], [30, 40], [10, 30]],
                              [])

    def test_complex_crop(self):
        # Crop generate GeometryCollection here
        exterior = [[0, 0], [0, 3], [5, 8], [5, 9], [5, 10], [0, 15], [10, 20], [0, 25], [20, 25], [20, 0]]
        interiors = [[[2, 2], [4, 4], [4, 2]]]

        poly = Polygon(exterior=row_col_list_to_points(exterior, flip_row_col_order=True),
                       interior=[row_col_list_to_points(interior, flip_row_col_order=True) for interior in interiors])

        crop_rect = Rectangle(0, 0, 30, 5)
        res_geoms = poly.crop(crop_rect)
        self.assertEqual(len(res_geoms), 3)
        self.assertPolyEquals(res_geoms[0],
                              [[0, 0], [5, 0], [5, 8], [0, 3]], interiors)

    def test_crop_by_border(self):
        exterior = [[10, 10], [40, 10], [40, 40], [10, 40]]
        interiors = [[[11, 11], [11, 20], [20, 11]], [[20, 20], [21, 20], [20, 21]]]
        poly = Polygon(exterior=row_col_list_to_points(exterior, flip_row_col_order=True),
                       interior=[row_col_list_to_points(interior, flip_row_col_order=True) for interior in interiors])

        crop_rect = Rectangle(0, 0, 100, 10)
        res_geoms = poly.crop(crop_rect)
        self.assertEqual(len(res_geoms), 0)

    def test_relative_crop(self):
        crop_rect = Rectangle(25, 0, 200, 200)
        res_geoms = self.poly.relative_crop(crop_rect)
        self.assertEqual(len(res_geoms), 1)
        crop = res_geoms[0]
        self.assertPolyEquals(crop, [[10, 0], [20, 0], [20, 5], [30, 5], [30, 0], [35, 0], [30, 15], [10, 5]], [])

    def test_rotate(self):
        imsize = (101, 101)
        rotator = ImageRotator(imsize=imsize, angle_degrees_ccw=-90)
        res_poly = self.poly.rotate(rotator)
        self.assertPolyEquals(res_poly,
                              [[90, 10], [90, 40], [60, 30], [70, 10]],
                              [[[80, 20], [80, 30], [70, 30], [70, 20]]])

    def test_resize(self):
        in_size = (100, 100)
        out_size = (200, 150)
        res_poly = self.poly.resize(in_size, out_size)
        self.assertPolyEquals(res_poly,
                              [[15, 20], [60, 20], [45, 80], [15, 60]],
                              [[[30, 40], [45, 40], [45, 60], [30, 60]]])

    def test_translate(self):
        drow = 5
        dcol = 10
        res_poly = self.poly.translate(drow, dcol)
        self.assertPolyEquals(res_poly,
                              [[20, 15], [50, 15], [40, 45], [20, 35]],
                              [[[30, 25], [40, 25], [40, 35], [30, 35]]])

    def test_fliplr(self):
        imsize = (100, 200)
        res_poly = self.poly.fliplr(imsize)
        self.assertPolyEquals(res_poly,
                              [[190, 10], [160, 10], [170, 40], [190, 30]],
                              [[[180, 20], [170, 20], [170, 30], [180, 30]]])

    def test_area(self):  # @TODO: only exterior area
        area = self.poly.area
        self.assertIsInstance(area, float)
        self.assertEqual(area, 650.0)

    def test_to_bbox(self):
        res_rect = self.poly.to_bbox()
        self.assertIsInstance(res_rect, Rectangle)
        self.assertEqual(res_rect.top, 10)
        self.assertEqual(res_rect.left, 10)
        self.assertEqual(res_rect.right, 40)
        self.assertEqual(res_rect.bottom, 40)

    def test_from_json(self):
        packed_obj = {
            'some_stuff': 'aaa',
            POINTS: {
                EXTERIOR: self.exterior,
                INTERIOR: self.interiors
            }
        }
        res_poly = Polygon.from_json(packed_obj)
        self.assertPolyEquals(res_poly, self.exterior, self.interiors)

    def test_to_json(self):
        res_obj = self.poly.to_json()
        expected_dict = {
            POINTS: {
                EXTERIOR: self.exterior,
                INTERIOR: self.interiors
            }
        }
        self.assertDictEqual(res_obj, expected_dict)

    def test_clone(self):
        res_poly = self.poly.clone()
        self.assertIsInstance(res_poly, Polygon)
        self.assertTrue(np.array_equal(res_poly.exterior_np, self.poly.exterior_np))
        self.assertTrue(np.array_equal(res_poly.interior_np, self.poly.interior_np))
        self.assertIsNot(res_poly, self.poly)

    def test_draw(self):
        def draw_mask(exterior_p, interior_p, h, w):
            mask = np.zeros((h, w), dtype=np.uint8)
            poly = Polygon(exterior=row_col_list_to_points(exterior_p),
                           interior=[row_col_list_to_points(interior) for interior in interior_p])
            poly.draw(mask, color=1)
            return mask

        # Test 1 - draw interior triangles
        exterior = [[0, 0], [0, 6], [4, 6], [4, 0]]
        interiors = [[[1, 1], [1, 3], [3, 1]],  # clockwise
                     [[1, 5], [3, 3], [3, 5]]]  # counterclockwise
        mask_1 = draw_mask(exterior, interiors, h=5, w=7)
        expected_mask = np.array([[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 1, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 1, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)

        self.assertTrue(np.array_equal(mask_1, expected_mask))

        # Test 1 - draw interior sandglass (bad poly case)
        exterior = [[0, 0], [0, 7], [7, 7], [7, 0]]
        interiors = [[[1, 1], [5, 5], [1, 5], [5, 1]]]  # sandglass
        mask_2 = draw_mask(exterior, interiors, h=7, w=7)
        expected_mask = np.array([[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 1, 1, 1, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 1, 1, 1, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(mask_2, expected_mask))

    def test_draw_contour(self):
        exterior = [[0, 0], [0, 6], [6, 6], [6, 0]]
        interiors = [[[2, 2], [2, 4], [4, 4], [4, 2]]]
        poly = Polygon(exterior=row_col_list_to_points(exterior),
                       interior=[row_col_list_to_points(interior) for interior in interiors])

        bitmap_1 = np.zeros((7, 7), dtype=np.uint8)
        poly.draw_contour(bitmap_1, color=1)

        expected_mask = np.array([[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 1, 1, 1, 0, 1],
                                  [1, 0, 1, 0, 1, 0, 1],
                                  [1, 0, 1, 1, 1, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(bitmap_1, expected_mask))

        # Extended test
        exterior = [[0, 0], [0, 6], [10, 6], [10, 0]]
        interiors = [[[1, 1], [1, 2], [2, 1]],
                     [[2, 4], [3, 5], [4, 4], [3, 3]],
                     [[6, 2], [8, 2], [8, 4], [6, 4]]]
        poly = Polygon(exterior=row_col_list_to_points(exterior),
                       interior=[row_col_list_to_points(interior) for interior in interiors])

        bitmap_2 = np.zeros((11, 7), dtype=np.uint8)
        poly.draw_contour(bitmap_2, color=1)

        expected_mask = np.array([[1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 0, 0, 0, 1],
                                  [1, 1, 0, 0, 1, 0, 1],
                                  [1, 0, 0, 1, 0, 1, 1],
                                  [1, 0, 0, 0, 1, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 1, 1, 1, 0, 1],
                                  [1, 0, 1, 0, 1, 0, 1],
                                  [1, 0, 1, 1, 1, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)

        self.assertTrue(np.array_equal(bitmap_2, expected_mask))


class BitmapTest(unittest.TestCase):
    def setUp(self):
        self.origin = PointLocation(0, 4)
        self.mask = np.array([[0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 1, 1, 0, 0],
                              [0, 1, 0, 1, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]], dtype=np.bool)
        self.mask_no_margins = self.mask[:, 1:-1]
        self.bitmap = Bitmap(data=self.mask, origin=self.origin)

    def assertBitmapEquals(self, bitmap, origin_row, origin_col, mask):
        self.assertIsInstance(bitmap, Bitmap),
        self.assertEqual(bitmap.origin.row, origin_row)
        self.assertEqual(bitmap.origin.col, origin_col)
        self.assertListEqual(bitmap.data.tolist(), mask.tolist())

    def test_rotate(self):
        in_size = (15, 15)
        rotator = ImageRotator(imsize=in_size, angle_degrees_ccw=90)
        res_bitmap = self.bitmap.rotate(rotator)
        expected_mask = np.array([[0, 0, 1, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0]], dtype=np.bool)
        self.assertListEqual(res_bitmap.data.tolist(), expected_mask.tolist())

    def test_empty_crop(self):
        crop_rect = Rectangle(0, 0, 4, 4)
        res_geoms = self.bitmap.crop(crop_rect)
        self.assertEqual(len(res_geoms), 0)

    def test_crop(self):  # @TODO: mb delete compress while cropping
        crop_rect = Rectangle(0, 0, 8, 8)
        res_geoms = self.bitmap.crop(crop_rect)
        self.assertEqual(len(res_geoms), 1)
        res_bitmap = res_geoms[0]
        res_mask = np.array([[0, 0, 1, 0],
                             [0, 1, 1, 1],
                             [1, 0, 1, 0],
                             [0, 0, 1, 0],
                             [0, 0, 1, 0],
                             [0, 0, 1, 0],
                             [0, 0, 1, 0]], dtype=np.bool)
        self.assertBitmapEquals(res_bitmap, 0, 5, res_mask)

    def test_translate(self):
        drows = 10
        dcols = 7
        res_bitmap = self.bitmap.translate(drows, dcols)
        self.assertBitmapEquals(res_bitmap, 10, 12, self.mask_no_margins)

    def test_area(self):
        area = self.bitmap.area
        self.assertIsInstance(area, float)
        self.assertEqual(area, 11)

    def test_to_bbox(self):
        res_rect = self.bitmap.to_bbox()
        self.assertIsInstance(res_rect, Rectangle)
        self.assertEqual(res_rect.top, 0)
        self.assertEqual(res_rect.left, 5)
        self.assertEqual(res_rect.right, 9)
        self.assertEqual(res_rect.bottom, 6)

    def test_to_contours(self):
        bitmap = Bitmap(data=np.array([[1, 1, 0, 1, 1, 1],
                                       [1, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 1, 1, 1],
                                       [1, 0, 0, 1, 0, 1],
                                       [1, 0, 0, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 1, 1, 1]], dtype=np.bool),
                        origin=PointLocation(10, 110))
        polygons = bitmap.to_contours()

        exteriors_points = [np.array([[10, 113], [14, 113], [14, 115], [10, 115]]),
                            np.array([[10, 110], [11, 110], [11, 111], [10, 111]])]

        interiors_points = [[],
                            [np.array([[13, 113], [12, 114], [13, 115], [14, 114]]),
                             np.array([[11, 113], [10, 114], [11, 115], [12, 114]])],
                           []]

        self.assertEqual(len(polygons), 2)
        for polygon, target_exterior, target_interiors in zip(polygons, exteriors_points, interiors_points):
            self.assertTrue(np.equal(polygon.exterior_np, target_exterior).all())
            self.assertTrue(all(np.equal(p_inter, t_inter)
                                for p_inter, t_inter in zip(polygon.interior_np, target_interiors)))
            json.dumps(polygon.to_json())
            self.assertIsInstance(polygon, Polygon)

    def test_from_json(self):
        packed_obj = {
            BITMAP: {
                DATA: 'eJzrDPBz5+WS4mJgYOD19HAJAtLsIMzIDCT/zTk6AUixBfiEuALp////L705/y6QxVgS5BfM4PDsRhqQI+j'
                      'p4hhSMSdZIGFDAkeiQIMDA7sVw125xatvACUZPF39XNY5JTQBADRqHJQ=',
                ORIGIN: [0, 1]
            }
        }
        bitmap = Bitmap.from_json(packed_obj)
        self.assertBitmapEquals(bitmap, 1, 1, self.mask_no_margins)

    def test_resize(self):
        in_size = (20, 20)
        out_size = (40, 40)
        res_bitmap = self.bitmap.resize(in_size, out_size)
        self.assertIsInstance(res_bitmap, Bitmap)

    def test_flipud(self):
        im_size = (20, 20)
        res_bitmap = self.bitmap.flipud(im_size)
        expected_mask = np.array([[0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [1, 0, 1, 0, 1],
                                  [0, 1, 1, 1, 0],
                                  [0, 0, 1, 0, 0]], dtype=np.bool)
        self.assertBitmapEquals(res_bitmap, 13, 5, expected_mask)

    def test_to_json(self):
        obj = self.bitmap.to_json()
        expected_dict = {
            BITMAP: {
                DATA: 'eJzrDPBz5+WS4mJgYOD19HAJAtKsQMzOyAwkf2WKrgVSbAE+Ia5A+v///0tvzr8LZDGWBPkFMzg8u5EG5Ah'
                      '6ujiGVMxJDkjQSIg4uIChkYEvjXHncvfAUqAkg6ern8s6p4QmAAVvHAE=',
                ORIGIN: [5, 0]
            }
        }
        self.assertDictEqual(obj, expected_dict)


class MultichannelBitmapTest(unittest.TestCase):
    def setUp(self):
        self.origin = PointLocation(row=0, col=4)
        self.data = np.array([[[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
                              [[0.6, 0.7], [0.8, 0.9], [1.0, 1.1]]], dtype=np.float64)
        self.bitmap = MultichannelBitmap(data=self.data, origin=self.origin)

    def assertMultichannelBitmapEquals(self, bitmap, origin_row, origin_col, data):
        self.assertIsInstance(bitmap, MultichannelBitmap),
        self.assertEqual(bitmap.origin.row, origin_row)
        self.assertEqual(bitmap.origin.col, origin_col)
        self.assertListEqual(bitmap.data.tolist(), data.tolist())

    def test_rotate(self):
        in_size = (15, 15)
        rotator = ImageRotator(imsize=in_size, angle_degrees_ccw=90)
        res_bitmap = self.bitmap.rotate(rotator)
        expected_data = np.array([[[0.4, 0.5], [1.0, 1.1]],
                                  [[0.2, 0.3], [0.8, 0.9]],
                                  [[0.0, 0.1], [0.6, 0.7]]], dtype=np.float64)
        # TODO origin
        self.assertListEqual(res_bitmap.data.tolist(), expected_data.tolist())

    def test_empty_crop(self):
        crop_rect = Rectangle(top=0, left=0, bottom=10, right=3)
        res_geoms = self.bitmap.crop(crop_rect)
        self.assertEqual(len(res_geoms), 0)

    def test_crop(self):
        crop_rect = Rectangle(top=1, left=0, bottom=10, right=4)
        res_geoms = self.bitmap.crop(crop_rect)
        self.assertEqual(len(res_geoms), 1)
        res_bitmap = res_geoms[0]
        res_data = np.array([[[0.6, 0.7]]], dtype=np.float64)
        self.assertMultichannelBitmapEquals(res_bitmap, 1, 4, res_data)

    def test_translate(self):
        drows = 10
        dcols = 7
        res_bitmap = self.bitmap.translate(drows, dcols)
        self.assertMultichannelBitmapEquals(res_bitmap, 10, 11, self.data)

    def test_area(self):
        area = self.bitmap.area
        self.assertEqual(area, 6)

    def test_to_bbox(self):
        res_rect = self.bitmap.to_bbox()
        self.assertIsInstance(res_rect, Rectangle)
        self.assertEqual(res_rect.top, 0)
        self.assertEqual(res_rect.left, 4)
        self.assertEqual(res_rect.right, 6)
        self.assertEqual(res_rect.bottom, 1)

    def test_resize(self):
        in_size = (20, 20)
        out_size = (40, 40)
        res_bitmap = self.bitmap.resize(in_size, out_size)
        expected_data = np.array([[[0.0, 0.1], [0.0, 0.1], [0.2, 0.3], [0.2, 0.3], [0.4, 0.5], [0.4, 0.5]],
                                  [[0.0, 0.1], [0.0, 0.1], [0.2, 0.3], [0.2, 0.3], [0.4, 0.5], [0.4, 0.5]],
                                  [[0.6, 0.7], [0.6, 0.7], [0.8, 0.9], [0.8, 0.9], [1.0, 1.1], [1.0, 1.1]],
                                  [[0.6, 0.7], [0.6, 0.7], [0.8, 0.9], [0.8, 0.9], [1.0, 1.1], [1.0, 1.1]]],
                                 dtype=np.float64)
        self.assertMultichannelBitmapEquals(res_bitmap, 0, 8, expected_data)

    def test_flipud(self):
        im_size = (20, 20)
        res_bitmap = self.bitmap.flipud(im_size)
        expected_data = np.array([[[0.6, 0.7], [0.8, 0.9], [1.0, 1.1]],
                                  [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]], dtype=np.float64)
        self.assertMultichannelBitmapEquals(res_bitmap, 18, 4, expected_data)

    def test_fliplr(self):
        im_size = (20, 20)
        res_bitmap = self.bitmap.fliplr(im_size)
        expected_data = np.array([[[0.4, 0.5], [0.2, 0.3], [0.0, 0.1]],
                                  [[1.0, 1.1], [0.8, 0.9], [0.6, 0.7]]], dtype=np.float64)
        self.assertMultichannelBitmapEquals(res_bitmap, 0, 13, expected_data)


if __name__ == '__main__':
    unittest.main()
