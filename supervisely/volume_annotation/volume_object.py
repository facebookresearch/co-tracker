# coding: utf-8

import uuid

from typing import Optional, Union
from numpy import ndarray

from supervisely.video_annotation.video_object import VideoObject
from supervisely.volume_annotation import volume_figure
from supervisely.volume_annotation.volume_tag_collection import VolumeTagCollection
from supervisely.geometry.mask_3d import Mask3D


class VolumeObject(VideoObject):
    """
    VolumeObject object for :class:`VolumeAnnotation<supervisely.volume_annotation.volume_annotation.VolumeAnnotation>`. :class:`VolumeObject<VolumeObject>` object is immutable.

    :param obj_class: VolumeObject :class:`class<supervisely.annotation.obj_class.ObjClass>`.
    :type obj_class: ObjClass
    :param tags: VolumeObject :class:`tags<supervisely.volume_annotation.volume_tag_collection.VolumeTagCollection>`.
    :type tags: VolumeTagCollection, optional
    :param key: The UUID key associated with the VolumeFigure.
    :type key: UUID, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which VolumeObject belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created VolumeObject.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when VolumeObject was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when VolumeObject was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :param mask_3d: Path for local geometry file, array with geometry data or Mask3D geometry object
    :type mask_3d: Union[str, ndarray, Mask3D], optional
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

    def __init__(
        self,
        obj_class,
        tags: Optional[VolumeTagCollection] = None,
        key: Optional[uuid.UUID] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        mask_3d: Optional[Union[Mask3D, ndarray, str]] = None,
    ):
        super().__init__(
            obj_class=obj_class,
            tags=tags,
            key=key,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

        if mask_3d is not None:
            if isinstance(mask_3d, str):
                self.figure = volume_figure.VolumeFigure(
                    self, Mask3D.create_from_file(mask_3d), labeler_login, updated_at, created_at
                )
            elif isinstance(mask_3d, ndarray):
                self.figure = volume_figure.VolumeFigure(
                    self, Mask3D(mask_3d), labeler_login, updated_at, created_at
                )
            elif isinstance(mask_3d, Mask3D):
                self.figure = volume_figure.VolumeFigure(
                    self, mask_3d, labeler_login, updated_at, created_at
                )
            else:
                raise TypeError("mask_3d type must be one of [Mask3D, ndarray, str]")
