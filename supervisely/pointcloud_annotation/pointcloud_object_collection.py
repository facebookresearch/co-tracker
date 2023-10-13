# coding: utf-8
from __future__ import annotations
from typing import List, Dict, Optional, Iterator
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap

class PointcloudObjectCollection(VideoObjectCollection):
    '''
    Collection with :class:`PointcloudObject<supervisely.pointcloud_annotation.pointcloud_object.PointcloudObject>` instances.
    :class:`PointcloudObjectCollection<PointcloudObjectCollection>` object is immutable.
    '''
    item_type = PointcloudObject

    def __iter__(self) -> Iterator[PointcloudObject]:
        return next(self)

    @classmethod
    def from_json(
        cls, 
        data: List[Dict], 
        project_meta: ProjectMeta, 
        key_id_map: Optional[KeyIdMap]=None
    ) -> PointcloudObjectCollection:
        """
        Convert a list of json dicts to PointcloudObjectCollection. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: List[dict]
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudObjectCollection object
        :rtype: :class:`PointcloudObjectCollection`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid_3d import Cuboid3d
            from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

            obj_collection_json = [
                {
                    "classTitle": "car",
                    "tags": []
                },
                {
                    "classTitle": "bus",
                    "tags": []
                }
            ]

            class_car = sly.ObjClass('car', Cuboid3d)
            class_bus = sly.ObjClass('bus', Cuboid3d)
            classes = sly.ObjClassCollection([class_car, class_bus])
            meta = sly.ProjectMeta(obj_classes=classes)

            pointcloud_obj_collection = sly.PointcloudObjectCollection.from_json(obj_collection_json, meta)
        """

        return super().from_json(data, project_meta, key_id_map=key_id_map)