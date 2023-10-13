# coding: utf-8

from __future__ import annotations
from typing import Tuple, Dict, Optional, List
import uuid
from uuid import UUID
from bidict import bidict

from supervisely.api.module_api import ApiField

from supervisely._utils import take_with_default
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.rectangle import Rectangle
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.video_annotation.constants import ID, KEY, OBJECT_ID, OBJECT_KEY, META
from supervisely.api.module_api import ApiField
from supervisely.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.video_annotation.video_object import VideoObject
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import (
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    CLASS_ID,
)


class OutOfImageBoundsException(Exception):
    pass


class VideoFigure:
    """
    VideoFigure object for :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`. :class:`VideoFigure<VideoFigure>` object is immutable.

    :param video_object: VideoObject object.
    :type video_object: VideoObject
    :param geometry: Label :class:`geometry<supervisely.geometry.geometry.Geometry>`.
    :type geometry: Geometry
    :param frame_index: Index of Frame to which VideoFigure belongs.
    :type frame_index: int
    :param key_id_map: KeyIdMap object.
    :type key_id_map: KeyIdMap, optional
    :param class_id: ID of :class:`VideoObject<VideoObject>` to which VideoFigure belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created VideoFigure.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when VideoFigure was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when VideoFigure was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        obj_class_car = sly.ObjClass('car', sly.Rectangle)
        video_obj_car = sly.VideoObject(obj_class_car)
        fr_index = 7
        geometry = sly.Rectangle(0, 0, 100, 100)
        video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)
        video_figure_car_json = video_figure_car.to_json()
        print(video_figure_car_json)
        # Output: {
        #     "key": "5e8afd2e26a54ab18154b355fa9665f8",
        #     "objectKey": "5860b7a5519b4de7b3d9c1720a40b38a",
        #     "geometryType": "rectangle",
        #     "geometry": {
        #         "points": {
        #             "exterior": [
        #                 [
        #                     0,
        #                     0
        #                 ],
        #                 [
        #                     100,
        #                     100
        #                 ]
        #             ],
        #             "interior": []
        #         }
        #     }
        # }
    """

    def __init__(
        self,
        video_object: VideoObject,
        geometry: Geometry,
        frame_index: int,
        key: Optional[UUID] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        self._video_object = video_object
        self._set_geometry_inplace(geometry)
        self._frame_index = frame_index
        self._key = take_with_default(key, uuid.uuid4())
        self.class_id = class_id
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at

    def _add_creation_info(self, d):
        if self.labeler_login is not None:
            d[LABELER_LOGIN] = self.labeler_login
        if self.updated_at is not None:
            d[UPDATED_AT] = self.updated_at
        if self.created_at is not None:
            d[CREATED_AT] = self.created_at

    def _set_geometry_inplace(self, geometry: Geometry) -> None:
        """
        Checks the given geometry for correctness. Raise error if given geometry type != geometry type of VideoObject class
        :param geometry: Geometry class object (Point, Rectangle etc)
        """
        self._geometry = geometry
        self._validate_geometry_type()
        self._validate_geometry()

    @property
    def video_object(self) -> VideoObject:
        """
        VideoObject of current VideoFigure.

        :return: VideoObject object
        :rtype: :class:`VideoObject<VideoObject>`
        :Usage example:

         .. code-block:: python

            video_obj_car = video_figure_car.video_object
            print(video_obj_car.to_json())
            # Output: {
            #     "key": "d573c6f081544e3da20022d932b259c1",
            #     "classTitle": "car",
            #     "tags": []
            # }
        """
        return self._video_object

    @property
    def parent_object(self) -> VideoObject:
        """
        VideoObject of current VideoFigure.

        :return: VideoObject object
        :rtype: :class:`VideoObject<VideoObject>`
        :Usage example:

         .. code-block:: python

            video_obj_car = video_figure_car.parent_object
            print(video_obj_car.to_json())
            # Output: {
            #     "key": "d573c6f081544e3da20022d932b259c1",
            #     "classTitle": "car",
            #     "tags": []
            # }
        """
        return self._video_object

    @property
    def geometry(self) -> Geometry:
        """
        Geometry of the current VideoFigure.

        :return: Geometry object
        :rtype: :class:`Geometry<supervisely.geometry>`
        :Usage example:

         .. code-block:: python

            geometry = video_figure_car.geometry
            print(geometry.to_json())
            # Output: {
            #     "points": {
            #         "exterior": [
            #             [
            #                 0,
            #                 0
            #             ],
            #             [
            #                 100,
            #                 100
            #             ]
            #         ],
            #         "interior": []
            #     }
            # }
        """
        return self._geometry

    @property
    def frame_index(self) -> int:
        """
        Frame index of the current VideoFigure.

        :return: Index of Frame to which VideoFigure belongs
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            fr_index = video_figure_car.frame_index
            print(fr_index) # 7
        """
        return self._frame_index

    def key(self) -> UUID:
        """
        Figure key.

        :return: Figure key.
        :rtype: UUID
        :Usage example:

         .. code-block:: python

            key = video_figure_car.key
            print(key) # 158e6cf4f4ac4c639fc6994aad127c16
        """

        return self._key

    def _validate_geometry(self):
        """
        Checks geometry of VideoFigure class object for correctness
        """
        self._geometry.validate(
            self.parent_object.obj_class.geometry_type.geometry_name(),
            self.parent_object.obj_class.geometry_config,
        )

    def _validate_geometry_type(self):
        """
        Raise error if given geometry type != geometry type of VideoObject class
        """
        if self.parent_object.obj_class.geometry_type != AnyGeometry:
            if type(self._geometry) is not self.parent_object.obj_class.geometry_type:
                raise RuntimeError(
                    "Input geometry type {!r} != geometry type of ObjClass {}".format(
                        type(self._geometry), self.parent_object.obj_class.geometry_type
                    )
                )

    def to_json(
        self, key_id_map: Optional[KeyIdMap] = None, save_meta: Optional[bool] = False
    ) -> Dict:
        """
        Convert the VideoFigure to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :param save_meta: Save frame index or not.
        :type save_meta: bool, optional
        :return: Json format as a dict
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            fr_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)
            video_figure_json = video_figure_car.to_json(save_meta=True)
            print(video_figure_json)
            # Output: {
            #     "key": "591d0511ba28462c8cd657691743359c",
            #     "objectKey": "e061bc50bd464c23a008b712d195570a",
            #     "geometryType": "rectangle",
            #     "geometry": {
            #         "points": {
            #             "exterior": [
            #                 [
            #                     0,
            #                     0
            #                 ],
            #                 [
            #                     100,
            #                     100
            #                 ]
            #             ],
            #             "interior": []
            #         }
            #     },
            #     "meta": {
            #         "frame": 7
            #     }
            # }
        """
        data_json = {
            KEY: self.key().hex,
            OBJECT_KEY: self.parent_object.key().hex,
            ApiField.GEOMETRY_TYPE: self.geometry.geometry_name(),
            ApiField.GEOMETRY: self.geometry.to_json(),
        }

        if key_id_map is not None:
            item_id = key_id_map.get_figure_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

            object_id = key_id_map.get_object_id(self.parent_object.key())
            if object_id is not None:
                data_json[OBJECT_ID] = object_id
        if save_meta is True:
            data_json[ApiField.META] = self.get_meta()

        self._add_creation_info(data_json)
        return data_json

    def get_meta(self) -> Dict[str, int]:
        """
        Get metadata for the video figure.

        :return: Dictionary with metadata for the video figure.
        :rtype: :py:class:`Dict[str, int]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            fr_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)

            print(video_figure_car.get_meta()) # {'frame': 7}
        """

        return {ApiField.FRAME: self.frame_index}

    @classmethod
    def from_json(
        cls,
        data: Dict,
        objects: VideoObjectCollection,
        frame_index: int,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> VideoFigure:
        """
        Convert a json dict to VideoFigure. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Dict in json format.
        :type data: :class:`dict`
        :param objects: VideoObjectCollection object.
        :type objects: VideoObjectCollection
        :param frame_index: Index of Frame to which VideoFigure belongs.
        :type frame_index: int
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :raises: :class:`RuntimeError`, if video object ID and video object key are None, if video object key and key_id_map are None, if video object with given id not found in key_id_map
        :return: VideoFigure object
        :rtype: :class:`VideoFigure`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Create VideoFigure from json we use data from example to_json(see above)
            new_video_figure = sly.VideoFigure.from_json(video_figure_json, sly.VideoObjectCollection([video_obj_car]), fr_index)
        """
        object_id = data.get(OBJECT_ID, None)
        object_key = None
        if OBJECT_KEY in data:
            object_key = uuid.UUID(data[OBJECT_KEY])

        if object_id is None and object_key is None:
            raise RuntimeError(
                "Figure can not be deserialized from json: object_id or object_key are not found"
            )

        if object_key is None:
            if key_id_map is None:
                raise RuntimeError("Figure can not be deserialized: key_id_map is None")
            object_key = key_id_map.get_object_key(object_id)
            if object_key is None:
                raise RuntimeError("Object with id={!r} not found in key_id_map".format(object_id))

        object = objects.get(object_key)
        if object is None:
            raise RuntimeError(
                "Figure can not be deserialized: corresponding object {!r} not found in ObjectsCollection".format(
                    object_key.hex
                )
            )

        shape_str = data[ApiField.GEOMETRY_TYPE]
        geometry_json = data[ApiField.GEOMETRY]

        shape = GET_GEOMETRY_FROM_STR(shape_str)
        geometry = shape.from_json(geometry_json)

        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_figure(key, data.get(ID, None))

        class_id = data.get(CLASS_ID, None)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)

        return cls(
            object,
            geometry,
            frame_index,
            key,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def clone(
        self,
        video_object: Optional[VideoObject] = None,
        geometry: Optional[Geometry] = None,
        frame_index: Optional[int] = None,
        key: Optional[UUID] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> VideoFigure:
        """
        Makes a copy of VideoFigure with new fields, if fields are given, otherwise it will use fields of the original VideoFigure.

        :param video_object: VideoObject object.
        :type video_object: VideoObject, optional
        :param geometry: Label :class:`geometry<supervisely.geometry.geometry.Geometry>`.
        :type geometry: Geometry, optional
        :param frame_index: Index of Frame to which VideoFigure belongs.
        :type frame_index: int, optional
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which VideoFigure belongs.
        :type class_id: int, optional
        :param labeler_login: Login of the user who created VideoFigure.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when VideoFigure was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when VideoFigure was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :return: VideoFigure object
        :rtype: :class:`VideoFigure`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            fr_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)

            obj_class_bus = sly.ObjClass('bus', sly.Rectangle)
            video_obj_bus = sly.VideoObject(obj_class_bus)
            fr_index_bus = 15
            geometry_bus = sly.Rectangle(0, 0, 500, 600)

            # Remember that VideoFigure object is immutable, and we need to assign new instance of VideoFigure to a new variable
            video_figure_bus = video_figure_car.clone(video_object=video_obj_bus, geometry=geometry_bus, frame_index=fr_index_bus)
            print(video_figure_bus.to_json())
            # Output: {
            #     "key": "c2f501e94f42483ebd202697608e8d26",
            #     "objectKey": "942c79137b4547c59193276317f73897",
            #     "geometryType": "rectangle",
            #     "geometry": {
            #         "points": {
            #             "exterior": [
            #                 [
            #                     0,
            #                     0
            #                 ],
            #                 [
            #                     600,
            #                     500
            #                 ]
            #             ],
            #             "interior": []
            #         }
            #     }
            # }
        """
        return self.__class__(
            video_object=take_with_default(video_object, self.parent_object),
            geometry=take_with_default(geometry, self.geometry),
            frame_index=take_with_default(frame_index, self.frame_index),
            key=take_with_default(key, self._key),
            class_id=take_with_default(class_id, self.class_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )

    def validate_bounds(
        self, img_size: Tuple[int, int], _auto_correct: Optional[bool] = False
    ) -> None:
        """
        Checks if given image with given size contains a figure.

        :param img_size: Size of the image (height, width).
        :type img_size: Tuple[int, int]
        :param _auto_correct: Correct the geometry of a shape if it is out of bounds or not.
        :type _auto_correct: bool, optional
        :raises: :class:`OutOfImageBoundsException<supervisely.video_annotation.video_figure.OutOfImageBoundsException>`, if figure is out of image bounds
        :return: None
        :rtype: :class:`NoneType`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            fr_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)

            im_size = (50, 200)
            video_figure_car.validate_bounds(im_size)
            # raise OutOfImageBoundsException("Figure is out of image bounds")
        """
        canvas_rect = Rectangle.from_size(img_size)
        if canvas_rect.contains(self.geometry.to_bbox()) is False:
            raise OutOfImageBoundsException("Figure is out of image bounds")

        if _auto_correct is True:
            geometries_after_crop = [
                cropped_geometry for cropped_geometry in self.geometry.crop(canvas_rect)
            ]
            if len(geometries_after_crop) != 1:
                raise OutOfImageBoundsException("Several geometries after crop")
            self._set_geometry_inplace(geometries_after_crop[0])
