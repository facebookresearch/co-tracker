# coding: utf-8

from typing import List
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.api.entity_annotation.object_api import ObjectApi
from supervisely.api.pointcloud.pointcloud_tag_api import PointcloudObjectTagApi


class PointcloudObjectApi(ObjectApi):
    """
    :class:`PointcloudObject<supervisely.pointcloud_annotation.pointcloud_object.PointcloudObject>` for :class:`PointcloudAnnotation<supervisely.pointcloud_annotation.pointcloud_annotation.PointcloudAnnotation>`.
    """

    def __init__(self, api):
        """
        :param api: Api class object
        """
        super().__init__(api)
        self.tag = PointcloudObjectTagApi(api)

    def append_bulk(
        self,
        pointcloud_id: int,
        objects: PointcloudObjectCollection,
        key_id_map: KeyIdMap = None,
    ) -> List[int]:
        """
        Add pointcloud objects to Annotation Objects.

        :param pointcloud_id: Point cloud ID in Supervidely.
        :type pointcloud_id: int
        :param objects: PointcloudAnnotation objects.
        :type objects: PointcloudObjectCollection
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: List of objects IDs
        :rtype: :class:`List[int]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.video_annotation.key_id_map import KeyIdMap

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 19442
            pointcloud_id = 19618685

            meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(meta_json)

            key_id_map = KeyIdMap()
            ann_info = api.pointcloud.annotation.download(pointcloud_id)
            ann = sly.PointcloudAnnotation.from_json(ann_info, project_meta, key_id_map)

            res = api.pointcloud.object.append_bulk(pointcloud_id, ann.objects, key_id_map)
            print(res)

            # Output: [5565915, 5565916, 5565917, 5565918, 5565919]
        """

        info = self._api.pointcloud.get_info_by_id(pointcloud_id)
        return self._append_bulk(
            self._api.pointcloud.tag,
            pointcloud_id,
            info.project_id,
            info.dataset_id,
            objects,
            key_id_map,
            is_pointcloud=True,
        )

    def append_to_dataset(
        self,
        dataset_id: int,
        objects: PointcloudObjectCollection,
        key_id_map: KeyIdMap = None,
    ) -> List[int]:
        """
        Add pointcloud objects to Dataset annotation objects.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :param objects: Pointcloud objects collection.
        :type objects: PointcloudObjectCollection
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: List of objects IDs
        :rtype: :class:`List[int]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudObjectCollection
            from supervisely.video_annotation.key_id_map import KeyIdMap

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 19442
            project = api.project.get_info_by_id(project_id)
            dataset = api.dataset.create(project.id, "demo_dataset", change_name_if_conflict=True)

            class_car = sly.ObjClass('car', sly.Cuboid)
            class_pedestrian = sly.ObjClass('pedestrian', sly.Cuboid)
            classes = sly.ObjClassCollection([class_car, class_pedestrian])
            project_meta = sly.ProjectMeta(classes)
            updated_meta = api.project.update_meta(project.id, project_meta.to_json())

            key_id_map = KeyIdMap()

            pedestrian_object = sly.PointcloudObject(class_pedestrian)
            car_object = sly.PointcloudObject(class_car)
            objects_collection = PointcloudObjectCollection([pedestrian_object, car_object])

            uploaded_objects_ids = api.pointcloud_episode.object.append_to_dataset(
                dataset.id,
                objects_collection,
                key_id_map,
            )
            print(uploaded_objects_ids)

            # Output: [5565920, 5565921, 5565922]
        """

        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        return self._append_bulk(
            self._api.pointcloud.tag,
            dataset_id,
            project_id,
            dataset_id,
            objects,
            key_id_map,
            is_pointcloud=True,
        )
