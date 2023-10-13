import unittest

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag import Tag, TagValueType
from supervisely.annotation.tag_collection import TagCollection
from supervisely.annotation.tag_meta import TagMeta
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.geometry.rectangle import Rectangle
from supervisely.metric.map_metric import MAPMetric, AP
from supervisely.project.project_meta import ProjectMeta

class PointTest(unittest.TestCase):
    def setUp(self):
        self._obj_class_gt = ObjClass(name='a', geometry_type=Rectangle)
        self._obj_class_pred = ObjClass(name='b', geometry_type=Rectangle)
        self._confidence_tag_meta = TagMeta(name='confidence', value_type=TagValueType.ANY_NUMBER)
        self._meta = ProjectMeta(
            obj_classes=ObjClassCollection([self._obj_class_gt, self._obj_class_pred]),
            tag_metas=TagMetaCollection([self._confidence_tag_meta]))

        # Will match self._pred_obj_1
        self._gt_obj_1 = Label(obj_class=self._obj_class_gt, geometry=Rectangle(0, 0, 10, 10))

        # Will match self._pred_obj_3
        self._gt_obj_2 = Label(obj_class=self._obj_class_gt, geometry=Rectangle(13, 13, 15, 15))

        # Will be a false negative
        self._gt_obj_3 = Label(obj_class=self._obj_class_gt, geometry=Rectangle(43, 43, 45, 45))

        # Will match self._gt_obj_1
        self._pred_obj_1 = Label(
            obj_class=self._obj_class_pred,
            geometry=Rectangle(0, 0, 9, 9),
            tags=TagCollection([Tag(meta=self._confidence_tag_meta, value=0.7)]))

        # Will be a false positive (self._pred_obj_1 has higher IoU).
        self._pred_obj_2 = Label(
            obj_class=self._obj_class_pred,
            geometry=Rectangle(0, 0, 8, 8),
            tags=TagCollection([Tag(meta=self._confidence_tag_meta, value=0.6)]))

        # Will match self._gt_obj_2
        self._pred_obj_3 = Label(
            obj_class=self._obj_class_pred,
            geometry=Rectangle(13, 13, 15, 15),
            tags=TagCollection([Tag(meta=self._confidence_tag_meta, value=0.1)]))

        # More false positives.
        self._pred_objs_fp = [
            Label(obj_class=self._obj_class_pred,
                  geometry=Rectangle(20, 20, 30, 30),
                  tags=TagCollection([Tag(meta=self._confidence_tag_meta, value=v / 100)]))
            for v in range(15, 85, 10)]

        self._metric_calculator = MAPMetric(class_mapping={'a': 'b'}, iou_threshold=0.5)

    def test_empty_gt(self):
        ann = Annotation(
            img_size=[100, 100], labels=[self._pred_obj_1, self._pred_obj_2, self._pred_obj_3] + self._pred_objs_fp)
        self._metric_calculator.add_pair(ann, ann)
        self.assertEqual(self._metric_calculator.get_total_metrics()[AP], 0)

    def test_empty_predictions(self):
        ann = Annotation(
            img_size=[100, 100], labels=[self._gt_obj_1, self._gt_obj_2, self._gt_obj_3])
        self._metric_calculator.add_pair(ann, ann)
        self.assertEqual(self._metric_calculator.get_total_metrics()[AP], 0)

    def test_with_matches(self):
        ann = Annotation(
            img_size=[100, 100],
            labels=[self._gt_obj_1, self._gt_obj_2, self._gt_obj_3,
                    self._pred_obj_1, self._pred_obj_2, self._pred_obj_3] + self._pred_objs_fp)
        self._metric_calculator.add_pair(ann, ann)

        # Sorted matches by confidence:
        # 0.75 - recall 0   precision 0
        # 0.7  + recall 1/3 precision 1/2
        # 0.65 - recall 1/3 precision 1/3
        # 0.6  - recall 1/3 precision 1/4
        # 0.55 - recall 1/3 precision 1/5
        # 0.45 - recall 1/3 precision 1/6
        # 0.35 - recall 1/3 precision 1/7
        # 0.25 - recall 1/3 precision 1/8
        # 0.15 - recall 2/3 precision 1/9
        # 0.1  + recall 2/3 precision 2/10

        # Recalls 0.7, 0.8, 0.9, 1.0 -> max precision 0.
        # Recalls 0.6, 0.5, 0.4      -> max precision 2/10
        # Recalls 0.3, 0.2, 0.1, 0.0 -> max precision 1/2
        expected_map = (4 * 0.0 + 3 * (2/10) + 4 * 1/2) / 11
        self.assertEqual(self._metric_calculator.get_total_metrics()[AP], expected_map)


if __name__ == '__main__':
    unittest.main()
