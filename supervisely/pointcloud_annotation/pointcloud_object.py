# coding: utf-8

# docs
import uuid
from typing import Optional

from supervisely.video_annotation.video_object import VideoObject
from supervisely.annotation.label import LabelJsonFields
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.constants import KEY, ID
from supervisely.pointcloud_annotation.pointcloud_tag_collection import PointcloudTagCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, CLASS_ID


class PointcloudObject(VideoObject):
    """
    PointcloudObject object for :class:`PointcloudAnnotation<supervisely.pointcloud_annotation.pointcloud_annotation.PointcloudAnnotation>`. :class:`PointcloudObject<PointcloudObject>` object is immutable.

    :param obj_class: :class:`class<supervisely.annotation.obj_class.ObjClass>` object.
    :type obj_class: ObjClass
    :param tags: :class:`tags<supervisely.video_annotation.video_tag_collection.VideoTagCollection>` object.
    :type tags: VideoTagCollection, optional
    :param key: KeyIdMap object.
    :type key: KeyIdMap, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which PointcloudObject belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created PointcloudObject.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when PointcloudObject was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when PointcloudObject was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.geometry.cuboid_3d import Cuboid3d

        obj_class_car = sly.ObjClass('car', Cuboid3d)
        pointcloud_obj_car = sly.PointcloudObject(obj_class_car)
        pointcloud_obj_car_json = pointcloud_obj_car.to_json()
        print(pointcloud_obj_car_json)
        # Output: {
        #     "key": "6b819f1840f84d669b32cdec225385f0",
        #     "classTitle": "car",
        #     "tags": []
        # }
    """

    @classmethod
    def from_json(cls, data, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None):
        """
        Convert PointcloudObject from json format to PointcloudObject class object. Raise error if object class name is not found in the given project meta.

        :param data: PointcloudObject in json format.
        :type data: dict
        :param project_meta: Project metadata.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudObject object.
        :rtype: PointcloudObject
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d

            key_id_map = KeyIdMap()
            project_id = 19441
            obj_class_car = sly.ObjClass('car', Cuboid3d)
            project_meta = sly.ProjectMeta([obj_class_car])
            api.project.update_meta(id=project_id, meta=project_meta)
            pointcloud_obj_car = sly.PointcloudObject(obj_class_car)
            pointcloud_obj_car_json = pointcloud_obj_car.to_json()
            new_pointcloud_obj_car = sly.PointcloudObject.from_json(
                data=pointcloud_obj_car_json,
                project_meta=project_meta,
                key_id_map=key_id_map
            )
            pprint(new_pointcloud_obj_car)
            # <supervisely.pointcloud_annotation.pointcloud_object.PointcloudObject object at 0x7f97e0ba8ed0>
        """

        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        if obj_class is None:
            raise RuntimeError(f'Failed to deserialize a object from JSON: class name {obj_class_name!r} '
                               f'was not found in the given project meta.')

        object_id = data.get(ID, None)

        existing_key = None
        if object_id is not None and key_id_map is not None:
            existing_key = key_id_map.get_object_key(object_id)
        json_key = uuid.UUID(data[KEY]) if KEY in data else None
        if (existing_key is not None) and (json_key is not None) and (existing_key != json_key):
            raise RuntimeError("Object id = {!r}: existing_key {!r} != json_key {!r}"
                               .format(object_id, existing_key, json_key))

        if existing_key is not None:
            key = existing_key
        elif json_key is not None:
            key = json_key
        else:
            key = uuid.uuid4()

        if key_id_map is not None and existing_key is None:
            key_id_map.add_object(key, object_id)

        class_id = data.get(CLASS_ID, None)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)

        return cls(obj_class=obj_class,
                   key=key,
                   tags=PointcloudTagCollection.from_json(data[LabelJsonFields.TAGS], project_meta.tag_metas),
                   class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)