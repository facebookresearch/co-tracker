# coding: utf-8

from supervisely.video_annotation.video_object import VideoObject


class VolumeObject(VideoObject):
    """
    VolumeObject object for :class:`VolumeAnnotation<supervisely.volume_annotation.volume_annotation.VolumeAnnotation>`. :class:`VolumeObject<VolumeObject>` object is immutable.

    :param obj_class: VolumeObject :class:`class<supervisely.annotation.obj_class.ObjClass>`.
    :type obj_class: ObjClass
    :param tags: VolumeObject :class:`tags<supervisely.volume_annotation.volume_tag_collection.VolumeTagCollection>`.
    :type tags: VolumeTagCollection, optional
    :param key: KeyIdMap object.
    :type key: KeyIdMap, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which VolumeObject belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created VolumeObject.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when VolumeObject was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when VolumeObject was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
        volume_obj_heart = sly.VolumeObject(obj_class_heart)
        volume_obj_heart_json = volume_obj_heart.to_json()
        print(volume_obj_heart_json)
        # Output: {
        #     "key": "6b819f1840f84d669b32cdec225385f0",
        #     "classTitle": "heart",
        #     "tags": []
        # }
    """

    pass
