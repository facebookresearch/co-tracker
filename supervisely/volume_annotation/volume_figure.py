# coding: utf-8
import uuid
from supervisely.video_annotation.video_figure import VideoFigure
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh
from supervisely.api.module_api import ApiField
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely._utils import take_with_default
from supervisely.volume_annotation.volume_object import VolumeObject
import supervisely.volume_annotation.constants as constants
from supervisely.volume_annotation.constants import ID, KEY, OBJECT_ID, OBJECT_KEY, META
from supervisely.geometry.constants import (
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    CLASS_ID,
)

from supervisely.volume_annotation.volume_object_collection import (
    VolumeObjectCollection,
)


class VolumeFigure(VideoFigure):
    """
    VolumeFigure object for :class:`VolumeAnnotation<supervisely.volume_annotation.volume_annotation.VolumeAnnotation>`. :class:`VolumeFigure<VolumeFigure>` object is immutable.

    :param volume_object: VolumeObject object.
    :type volume_object: VolumeObject
    :param geometry: Geometry object
    :type geometry: :class:`Geometry<supervisely.geometry>`
    :param plane_name: Name of the volume plane.
    :type plane_name: str
    :param slice_index: Index of slice to which VolumeFigure belongs.
    :type slice_index: int
    :param key: KeyIdMap object.
    :type key: KeyIdMap, optional
    :param class_id: ID of :class:`VolumeObject<VolumeObject>` to which VolumeFigure belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created VolumeFigure.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when VolumeFigure was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when VolumeFigure was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
        volume_obj_heart = sly.VolumeObject(obj_class_heart)
        slice_index = 7
        plane_name = "axial"
        geometry = sly.Rectangle(0, 0, 100, 100)
        volume_figure_heart = sly.VolumeFigure(volume_obj_heart, geometry, plane_name, slice_index)
        volume_figure_heart_json = volume_figure_heart.to_json()
        print(volume_figure_heart_json)
        # Output: {
        #     "geometry": {
        #         "points": {
        #         "exterior": [
        #             [0, 0],
        #             [100, 100]
        #         ],
        #         "interior": []
        #         }
        #     },
        #     "geometryType": "rectangle",
        #     "key": "158e6cf4f4ac4c639fc6994aad127c16",
        #     "meta": {
        #         "normal": { "x": 0, "y": 0, "z": 1 },
        #         "planeName": "axial",
        #         "sliceIndex": 7
        #     },
        #     "objectKey": "bf63ffe342e949899d3ddcb6b0f73f54"
        # }
    """

    def __init__(
        self,
        volume_object,
        geometry,
        plane_name,
        slice_index,
        key=None,
        class_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        super().__init__(
            video_object=volume_object,
            geometry=geometry,
            frame_index=slice_index,
            key=key,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        from supervisely.volume_annotation.plane import Plane

        Plane.validate_name(plane_name)
        self._plane_name = plane_name
        self._slice_index = slice_index

    @property
    def volume_object(self) -> VolumeObject:
        """
        Get a parent VolumeObject object of volume figure.

        :return: Parent VolumeObject object of volume figure.
        :rtype: VolumeObject
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            volume_figure_heart = sly.VolumeFigure(
                volume_obj_heart,
                geometry=sly.Rectangle(0, 0, 100, 100),
                plane_name="axial",
                slice_index=7
            )

            print(volume_figure_heart.parent_object)
            # Output:
            # <supervisely.volume_annotation.volume_object.VolumeObject object at 0x7f95f0950b50>
        """

        return self._video_object

    @property
    def video_object(self):
        """Property "video_object" is only available for videos."""
        raise NotImplementedError('Property "video_object" is only available for videos')

    @property
    def parent_object(self) -> VolumeObject:
        """
        Get a parent VolumeObject object of volume figure.

        :return: VolumeObject object
        :rtype: VolumeObject
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            volume_figure_heart = sly.VolumeFigure(
                volume_obj_heart,
                geometry=sly.Rectangle(0, 0, 100, 100),
                plane_name="axial",
                slice_index=7
            )

            print(volume_figure_heart.parent_object)
            # Output:
            # <supervisely.volume_annotation.volume_object.VolumeObject object at 0x7f786a3f8bd0>
        """

        return self.volume_object

    @property
    def frame_index(self):
        """Property "frame_index" is only available for videos."""
        raise NotImplementedError('Property "frame_index" is only available for videos')

    @property
    def slice_index(self):
        """
        Get a slice index of volume figure.

        :return: :py:class:`Slice<supervisely.volume_annotation.slice.Slice>` index of volume figure.
        :rtype: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            volume_figure_heart = sly.VolumeFigure(
                volume_obj_heart,
                geometry=sly.Rectangle(0, 0, 100, 100),
                plane_name="axial",
                slice_index=7
            )

            print(volume_figure_heart.slice_index)
            # Output: 7
        """

        return self._slice_index

    @property
    def plane_name(self):
        """
        Get a plane name of volume figure.

        :return: Plane name of volume figure.
        :rtype: str
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            volume_figure_heart = sly.VolumeFigure(
                volume_obj_heart,
                geometry=sly.Rectangle(0, 0, 100, 100),
                plane_name="axial",
                slice_index=7
            )

            print(volume_figure_heart.plane_name)
            # Output: axial
        """

        return self._plane_name

    @property
    def normal(self):
        """
        Get a normal vector associated with a plane name.

        :return: Dictionary with normal vector associated with a plane name.
        :rtype: dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            volume_figure_heart = sly.VolumeFigure(
                volume_obj_heart,
                geometry=sly.Rectangle(0, 0, 100, 100),
                plane_name="axial",
                slice_index=7
            )

            print(volume_figure_heart.normal)
            # Output: {'x': 0, 'y': 0, 'z': 1}
        """

        from supervisely.volume_annotation.plane import Plane

        return Plane.get_normal(self.plane_name)

    def _validate_geometry_type(self):
        if (
            self.parent_object.obj_class.geometry_type != AnyGeometry
            and type(self._geometry) != ClosedSurfaceMesh
        ):
            if type(self._geometry) is not self.parent_object.obj_class.geometry_type:
                raise RuntimeError(
                    "Input geometry type {!r} != geometry type of ObjClass {}".format(
                        type(self._geometry), self.parent_object.obj_class.geometry_type
                    )
                )

    def _validate_geometry(self):
        if type(self._geometry) == ClosedSurfaceMesh:
            return
        super()._validate_geometry()

    def validate_bounds(self, img_size, _auto_correct=False):
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

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            slice_index = 7
            plane_name = "axial"
            geometry = sly.Rectangle(0, 0, 100, 100)
            volume_figure_heart = sly.VolumeFigure(volume_obj_heart, geometry, plane_name, slice_index)

            im_size = (50, 200)
            volume_figure_heart.validate_bounds(im_size)
            # raise OutOfImageBoundsException("Figure is out of image bounds")
        """

        if type(self._geometry) == ClosedSurfaceMesh:
            return
        super().validate_bounds(img_size, _auto_correct)

    def clone(
        self,
        volume_object=None,
        geometry=None,
        plane_name=None,
        slice_index=None,
        key=None,
        class_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        """
        Makes a copy of VolumeFigure with new fields, if fields are given, otherwise it will use fields of the original VolumeFigure.

        :param volume_object: VolumeObject object.
        :type volume_object: VolumeObject, optional
        :param geometry: Geometry object.
        :type geometry: :class:`Geometry<supervisely.geometry>`
        :param plane_name: Name of the volume plane.
        :type plane_name: str, optional
        :param slice_index: Index of slice to which VolumeFigure belongs.
        :type slice_index: int, optional
        :param key: KeyIdMap object.
        :type key: KeyIdMap, optional
        :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which VolumeFigure belongs.
        :type class_id: int, optional
        :param labeler_login: Login of the user who created VolumeFigure.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when VolumeFigure was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when VolumeFigure was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :return: VolumeFigure object
        :rtype: :class:`VolumeFigure`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            slice_index = 7
            plane_name = "axial"
            geometry = sly.Rectangle(0, 0, 100, 100)
            volume_figure_heart = sly.VolumeFigure(volume_obj_heart, geometry, plane_name, slice_index)

            obj_class_lang = sly.ObjClass('lang', sly.Rectangle)
            volume_obj_lang = sly.VolumeObject(obj_class_lang)
            slice_index_lang = 15
            geometry_lang = sly.Rectangle(0, 0, 500, 600)

            # Remember that VolumeFigure object is immutable, and we need to assign new instance of VolumeFigure to a new variable
            volume_figure_lang = volume_figure_heart.clone(volume_object=volume_obj_lang, geometry=geometry_lang, slice_index=slice_index_lang)
            print(volume_figure_lang.to_json())
            # Output: {
            #     "geometry": {
            #         "points": {
            #         "exterior": [
            #             [0, 0],
            #             [600, 500]
            #         ],
            #         "interior": []
            #         }
            #     },
            #     "geometryType": "rectangle",
            #     "key": "2974165267224bf6b677e17ca2304b04",
            #     "meta": {
            #         "normal": { "x": 0, "y": 0, "z": 1 },
            #         "planeName": "axial",
            #         "sliceIndex": 15
            #     },
            #     "objectKey": "dafe3adaacad474ba5163ecebcc57cd0"
            # }

        """

        return self.__class__(
            volume_object=take_with_default(volume_object, self.parent_object),
            geometry=take_with_default(geometry, self.geometry),
            plane_name=take_with_default(plane_name, self.plane_name),
            slice_index=take_with_default(slice_index, self.slice_index),
            key=take_with_default(key, self._key),
            class_id=take_with_default(class_id, self.class_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )

    def get_meta(self):
        """
        Get a dictionary with metadata associated with volume figure.

        :return: Dictionary with metadata associated with volume figure.
        :rtype: dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            volume_figure_heart = sly.VolumeFigure(
                volume_obj_heart,
                geometry=sly.Rectangle(0, 0, 100, 100),
                plane_name="axial",
                slice_index=7
            )

            print(volume_figure_heart.get_meta())
            # {'sliceIndex': 7, 'planeName': 'axial', 'normal': {'x': 0, 'y': 0, 'z': 1}}
        """

        return {
            constants.SLICE_INDEX: self.slice_index,
            constants.PLANE_NAME: self.plane_name,
            constants.NORMAL: self.normal,
        }

    @classmethod
    def from_json(
        cls,
        data,
        objects: VolumeObjectCollection,
        plane_name,
        slice_index,
        key_id_map: KeyIdMap = None,
    ):
        """
        Convert a json dict to VolumeFigure. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Dict in json format.
        :type data: dict
        :param objects: VolumeObjectCollection object.
        :type objects: VolumeObjectCollection
        :param plane_name: Name of the volume plane.
        :type plane_name: str
        :param slice_index: Index of slice to which VolumeFigure belongs.
        :type slice_index: int
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :raises: :class:`RuntimeError`, if volume object ID and volume object key are None, if volume object key and key_id_map are None, if volume object with given id not found in key_id_map
        :return: VolumeFigure object
        :rtype: :class:`VolumeFigure`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Create VolumeFigure from json we use data from example to_json(see above)
            new_volume_figure = sly.VolumeFigure.from_json(
                data=volume_figure_json,
                objects=sly.VolumeObjectCollection([volume_obj_heart]),
                plane_name="axial",
                slice_index=7
            )
        """

        # @#TODO: copypaste from video figure, add base class and refactor copypaste later
        # _video_figure = super().from_json(data, objects, slice_index, key_id_map)

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

        volume_object = objects.get(object_key)
        if volume_object is None:
            raise RuntimeError(
                "Figure can not be deserialized: corresponding object {!r} not found in ObjectsCollection".format(
                    object_key.hex
                )
            )

        shape_str = data[ApiField.GEOMETRY_TYPE]
        shape = GET_GEOMETRY_FROM_STR(shape_str)
        if shape == ClosedSurfaceMesh:
            geometry_json = data
        else:
            geometry_json = data[ApiField.GEOMETRY]
        geometry = shape.from_json(geometry_json)

        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_figure(key, data.get(ID, None))

        class_id = data.get(CLASS_ID, None)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)

        return cls(
            volume_object=volume_object,
            geometry=geometry,
            plane_name=plane_name,
            slice_index=slice_index,
            key=key,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def to_json(self, key_id_map=None, save_meta=True):
        """
        Convert the VolumeFigure to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :param save_meta: Save frame index or not.
        :type save_meta: bool, optional
        :return: Json format as a dict
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            fr_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            volume_figure_heart = sly.VolumeFigure(volume_obj_heart, geometry, fr_index)
            volume_figure_json = volume_figure_heart.to_json(save_meta=True)
            print(volume_figure_json)
            # Output: {
            #     "geometry": {
            #         "points": {
            #         "exterior": [
            #             [0, 0],
            #             [100, 100]
            #         ],
            #         "interior": []
            #         }
            #     },
            #     "geometryType": "rectangle",
            #     "key": "158e6cf4f4ac4c639fc6994aad127c16",
            #     "meta": {
            #         "normal": { "x": 0, "y": 0, "z": 1 },
            #         "planeName": "axial",
            #         "sliceIndex": 7
            #     },
            #     "objectKey": "bf63ffe342e949899d3ddcb6b0f73f54"
            # }
        """

        json_data = super().to_json(key_id_map, save_meta)
        if type(self._geometry) == ClosedSurfaceMesh:
            json_data.pop(ApiField.GEOMETRY)
            json_data.pop(ApiField.META)
        return json_data
