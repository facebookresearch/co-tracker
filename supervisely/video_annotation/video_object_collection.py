# coding: utf-8
from __future__ import annotations
from typing import List, Dict, Optional, Any, Iterator

from supervisely.video_annotation.video_object import VideoObject
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap


class VideoObjectCollection(KeyIndexedCollection):
    """
    Collection with :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` instances. :class:`VideoObjectCollection<VideoObjectCollection>` object is immutable.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create two VideoObjects for collection
        class_car = sly.ObjClass('car', sly.Rectangle)
        obj_car = sly.VideoObject(class_car)
        class_bus = sly.ObjClass('bus', sly.Rectangle)
        obj_bus = sly.VideoObject(class_bus)

        # Create VideoObjectCollection
        obj_collection = sly.VideoObjectCollection([obj_car, obj_bus])
        obj_collection_json = obj_collection.to_json()
        print(obj_collection_json)
        # Output: [
        #     {
        #         "key": "773300c59fde4707887068d555269ba5",
        #         "classTitle": "car",
        #         "tags": []
        #     },
        #     {
        #         "key": "8257371497d2402cadabc690f796b1d1",
        #         "classTitle": "bus",
        #         "tags": []
        #     }
        # ]

        # Add item to VideoObjectCollection
        class_truck = sly.ObjClass('truck', sly.Rectangle)
        obj_truck = sly.VideoObject(class_truck)
        # Remember that VideoObjectCollection is immutable, and we need to assign new instance of VideoObjectCollection to a new variable
        new_obj_collection = obj_collection.add(obj_truck)
        new_obj_collection_json = new_obj_collection.to_json()
        print(new_obj_collection_json)
        # Output: [
        #     {
        #         "key": "1b62882f180f49ae96744e13fba26d02",
        #         "classTitle": "car",
        #         "tags": []
        #     },
        #     {
        #         "key": "0c95107fce6c4ac1b42132e83c1a6b3b",
        #         "classTitle": "bus",
        #         "tags": []
        #     },
        #     {
        #         "key": "660d8572e20c4101b4cc7edf7f04b090",
        #         "classTitle": "truck",
        #         "tags": []
        #     }
        # ]

        # You can also add multiple items to collection
        class_truck = sly.ObjClass('truck', sly.Rectangle)
        obj_truck = sly.VideoObject(class_truck)
        class_train = sly.ObjClass('train', sly.Rectangle)
        obj_train = sly.VideoObject(class_train)
        # Remember that VideoObjectCollection is immutable, and we need to assign new instance of VideoObjectCollection to a new variable
        new_obj_collection = obj_collection.add_items([obj_truck, obj_train])
        new_obj_collection_json = new_obj_collection.to_json()
        print(new_obj_collection_json)
        # Output: [
        #     {
        #         "key": "892c62aae48b495687fe8b33ce8ebe96",
        #         "classTitle": "car",
        #         "tags": []
        #     },
        #     {
        #         "key": "db0f64a7b60447769374943077e57679",
        #         "classTitle": "bus",
        #         "tags": []
        #     },
        #     {
        #         "key": "72c24a107f344ef68d4fe8e93dff8184",
        #         "classTitle": "truck",
        #         "tags": []
        #     },
        #     {
        #         "key": "d20738a608234152b43b1c23b7958c47",
        #         "classTitle": "train",
        #         "tags": []
        #     }
        # ]

        # Intersection, finds intersection of given list of instances with collection items

        obj_intersections = obj_collection.intersection([obj_car])
        obj_intersections_json = obj_intersections.to_json()
        print(obj_intersections_json)
        # Output: [
        #     {
        #         "key": "5fae14a4a42c42a29904a887442162c9",
        #         "classTitle": "car",
        #         "tags": []
        #     }
        # ]

        # Difference, finds difference between collection and given list of VideoObject

        obj_diff = obj_collection.difference([obj_car])
        obj_diff_json = obj_diff.to_json()
        print(obj_diff_json)
        # Output: [
        #     {
        #         "key": "b4365bbec0314de58e20267735a39164",
        #         "classTitle": "bus",
        #         "tags": []
        #     }
        # ]

        # Merge, merges collection and given list of VideoObjectCollection

        class_truck = sly.ObjClass('truck', sly.Rectangle)
        obj_truck = sly.VideoObject(class_truck)
        class_train = sly.ObjClass('train', sly.Rectangle)
        obj_train = sly.VideoObject(class_train)
        over_collection = sly.VideoObjectCollection([obj_truck, obj_train])
        # Merge
        merge_collection = obj_collection.merge(over_collection)
        merge_collection_json = merge_collection.to_json()
        print(merge_collection_json)
        # Output: [
        #     {
        #         "key": "8e6f9f05f0954193bdb2005b9cd6e20b",
        #         "classTitle": "truck",
        #         "tags": []
        #     },
        #     {
        #         "key": "a010f91a5e7646bdb8e06dfd6fa50f30",
        #         "classTitle": "train",
        #         "tags": []
        #     },
        #     {
        #         "key": "8440e915a94640b0be279c4fa293553b",
        #         "classTitle": "car",
        #         "tags": []
        #     },
        #     {
        #         "key": "e03e7dd2a2d14854945b6ee7bf7c72e3",
        #         "classTitle": "bus",
        #         "tags": []
        #     }
        # ]
    """
    item_type = VideoObject

    def __iter__(self) -> Iterator[VideoObject]:
        return next(self)

    def to_json(self, key_id_map: Optional[KeyIdMap]=None) -> List[Dict]:
        """
        Convert the VideoObjectCollection to a list of json dicts. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: List of dicts in json format
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            class_car = sly.ObjClass('car', sly.Rectangle)
            obj_car = sly.VideoObject(class_car)
            class_bus = sly.ObjClass('bus', sly.Rectangle)
            obj_bus = sly.VideoObject(class_bus)
            obj_collection = sly.VideoObjectCollection([obj_car, obj_bus])
            obj_collection_json = obj_collection.to_json()
            print(obj_collection_json)
            # Output: [
            #     {
            #         "key": "773300c59fde4707887068d555269ba5",
            #         "classTitle": "car",
            #         "tags": []
            #     },
            #     {
            #         "key": "8257371497d2402cadabc690f796b1d1",
            #         "classTitle": "bus",
            #         "tags": []
            #     }
            # ]
        """
        return [item.to_json(key_id_map) for item in self]

    @classmethod
    def from_json(cls, data: List[Dict], project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap]=None) -> VideoObjectCollection:
        """
        Convert a list of json dicts to VideoObjectCollection. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: List[dict]
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: VideoObjectCollection object
        :rtype: :class:`VideoObjectCollection`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

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

            class_car = sly.ObjClass('car', sly.Rectangle)
            class_bus = sly.ObjClass('bus', sly.Rectangle)
            classes = sly.ObjClassCollection([class_car, class_bus])
            meta = sly.ProjectMeta(obj_classes=classes)

            video_obj_collection = sly.VideoObjectCollection.from_json(obj_collection_json, meta)
        """
        objects = [cls.item_type.from_json(video_object_json, project_meta, key_id_map) for video_object_json in data]
        return cls(objects)

    def __str__(self):
        return 'Objects:\n' + super(VideoObjectCollection, self).__str__()