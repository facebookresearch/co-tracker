# coding: utf-8
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.volume_annotation.volume_object import VolumeObject


class VolumeObjectCollection(VideoObjectCollection):
    """
    Collection with :class:`VolumeObject<supervisely.volume_annotation.volume_object.VolumeObject>` instances. :class:`VolumeObjectCollection<VolumeObjectCollection>` object is immutable.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create two VolumeObjects for collection
        class_heart = sly.ObjClass('heart', sly.Rectangle)
        obj_heart = sly.VolumeObject(class_heart)
        class_lang = sly.ObjClass('lang', sly.Rectangle)
        obj_lang = sly.VolumeObject(class_lang)

        # Create VolumeObjectCollection
        obj_collection = sly.VolumeObjectCollection([obj_heart, obj_lang])
        obj_collection_json = obj_collection.to_json()
        print(obj_collection_json)
        # Output: [
        #     {
        #         "key": "773300c59fde4707887068d555269ba5",
        #         "classTitle": "heart",
        #         "tags": []
        #     },
        #     {
        #         "key": "8257371497d2402cadabc690f796b1d1",
        #         "classTitle": "lang",
        #         "tags": []
        #     }
        # ]

        # Add item to VolumeObjectCollection
        class_arm = sly.ObjClass('arm', sly.Rectangle)
        obj_arm = sly.VolumeObject(class_arm)
        # Remember that VolumeObjectCollection is immutable, and we need to assign new instance of VolumeObjectCollection to a new variable
        new_obj_collection = obj_collection.add(obj_arm)
        new_obj_collection_json = new_obj_collection.to_json()
        print(new_obj_collection_json)
        # Output: [
        #     {
        #         "key": "1b62882f180f49ae96744e13fba26d02",
        #         "classTitle": "heart",
        #         "tags": []
        #     },
        #     {
        #         "key": "0c95107fce6c4ac1b42132e83c1a6b3b",
        #         "classTitle": "lang",
        #         "tags": []
        #     },
        #     {
        #         "key": "660d8572e20c4101b4cc7edf7f04b090",
        #         "classTitle": "arm",
        #         "tags": []
        #     }
        # ]

        # You can also add multiple items to collection
        class_arm = sly.ObjClass('arm', sly.Rectangle)
        obj_arm = sly.VolumeObject(class_arm)
        class_train = sly.ObjClass('train', sly.Rectangle)
        obj_train = sly.VolumeObject(class_train)
        # Remember that VolumeObjectCollection is immutable, and we need to assign new instance of VolumeObjectCollection to a new variable
        new_obj_collection = obj_collection.add_items([obj_arm, obj_train])
        new_obj_collection_json = new_obj_collection.to_json()
        print(new_obj_collection_json)
        # Output: [
        #     {
        #         "key": "892c62aae48b495687fe8b33ce8ebe96",
        #         "classTitle": "heart",
        #         "tags": []
        #     },
        #     {
        #         "key": "db0f64a7b60447769374943077e57679",
        #         "classTitle": "lang",
        #         "tags": []
        #     },
        #     {
        #         "key": "72c24a107f344ef68d4fe8e93dff8184",
        #         "classTitle": "arm",
        #         "tags": []
        #     },
        #     {
        #         "key": "d20738a608234152b43b1c23b7958c47",
        #         "classTitle": "train",
        #         "tags": []
        #     }
        # ]

        # Intersection, finds intersection of given list of instances with collection items

        obj_intersections = obj_collection.intersection([obj_heart])
        obj_intersections_json = obj_intersections.to_json()
        print(obj_intersections_json)
        # Output: [
        #     {
        #         "key": "5fae14a4a42c42a29904a887442162c9",
        #         "classTitle": "heart",
        #         "tags": []
        #     }
        # ]

        # Difference, finds difference between collection and given list of VolumeObject

        obj_diff = obj_collection.difference([obj_heart])
        obj_diff_json = obj_diff.to_json()
        print(obj_diff_json)
        # Output: [
        #     {
        #         "key": "b4365bbec0314de58e20267735a39164",
        #         "classTitle": "lang",
        #         "tags": []
        #     }
        # ]

        # Merge, merges collection and given list of VolumeObjectCollection

        class_arm = sly.ObjClass('arm', sly.Rectangle)
        obj_arm = sly.VolumeObject(class_arm)
        class_train = sly.ObjClass('train', sly.Rectangle)
        obj_train = sly.VolumeObject(class_train)
        over_collection = sly.VolumeObjectCollection([obj_arm, obj_train])
        # Merge
        merge_collection = obj_collection.merge(over_collection)
        merge_collection_json = merge_collection.to_json()
        print(merge_collection_json)
        # Output: [
        #     {
        #         "key": "8e6f9f05f0954193bdb2005b9cd6e20b",
        #         "classTitle": "arm",
        #         "tags": []
        #     },
        #     {
        #         "key": "a010f91a5e7646bdb8e06dfd6fa50f30",
        #         "classTitle": "train",
        #         "tags": []
        #     },
        #     {
        #         "key": "8440e915a94640b0be279c4fa293553b",
        #         "classTitle": "heart",
        #         "tags": []
        #     },
        #     {
        #         "key": "e03e7dd2a2d14854945b6ee7bf7c72e3",
        #         "classTitle": "lang",
        #         "tags": []
        #     }
        # ]
    """

    item_type = VolumeObject
