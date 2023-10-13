# coding: utf-8

from copy import deepcopy
from typing import List, Optional, Tuple, Union
from supervisely._utils import take_with_default, validate_img_size
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation import constants
from supervisely.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely.volume_annotation.slice import Slice
from supervisely.sly_logger import logger

# example:
# "volumeMeta": {
#     "ACS": "RAS",
#     "intensity": { "max": 3071, "min": -3024 },
#     "windowWidth": 6095,
#     "rescaleSlope": 1,
#     "windowCenter": 23.5,
#     "channelsCount": 1,
#     "dimensionsIJK": { "x": 512, "y": 512, "z": 139 },
#     "IJK2WorldMatrix": [
#         0.7617189884185793, 0, 0, -194.238403081894, 0, 0.7617189884185793, 0,
#         -217.5384061336518, 0, 0, 2.5, -347.7500000000001, 0, 0, 0, 1
#     ],
#     "rescaleIntercept": 0
# },


class Plane(FrameCollection):
    """
    A class representing a plane in medical image data.

    :param plane_name: Name of the plane, should be one of "sagittal", "coronal", "axial", or None for spatial figures.
    :type plane_name: Union[str, None]
    :param img_size: Size of the plane image
    :type img_size: Optional[Union[Tuple[int, int], None]]
    :param slices_count: Number of slices in the plane.
    :type slices_count: Optional[Union[int, None]]
    :param items: List of :py:class:`Slice<supervisely.volume_annotation.slice.Slice>` objects representing the slices in the plane.
    :type items: Oprional[List[:py:class:`Slice<supervisely.volume_annotation.slice.Slice>`]]
    :param volume_meta: Metadata of the volume.
    :type volume_meta: Optional[dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.volume_annotation.plane import Plane
        path = "/Users/almaz/Downloads/my volumes/ds11111/Demo volumes_ds1_CTChest.nrrd"
        vol, meta  = sly.volume.read_nrrd_serie_volume(path)

        plane = Plane(
            sly.Plane.AXIAL,
            volume_meta=meta,
        )
        print(plane.name) # axial
    """

    item_type = Slice

    SAGITTAL = "sagittal"
    """Sagittal plane of the volume."""

    CORONAL = "coronal"
    """Coronal plane of the volume."""

    AXIAL = "axial"
    """Axial plane of the volume."""

    _valid_names = [SAGITTAL, CORONAL, AXIAL, None]  # None for spatial figures

    @staticmethod
    def validate_name(name: Union[str, None]):
        """
        Validates if the given plane name is valid.

        :param name: Name of the plane.
        :type name: Union[str, None]

        :raises ValueError: If `name` is not one of "sagittal", "coronal", "axial", or None.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.volume_annotation.plane import Plane
            plane_name_1 = "axial"
            Plane.validate_name(plane_name_1)

            plane_name_2 = "xy"
            Plane.validate_name(plane_name_2)
            # ValueError: Unknown plane xy, valid names are ['sagittal', 'coronal', 'axial', None]
        """

        if name not in Plane._valid_names:
            raise ValueError(f"Unknown plane {name}, valid names are {Plane._valid_names}")

    def __init__(
        self,
        plane_name: Union[str, None],
        img_size: Union[tuple, list] = None,
        slices_count: int = None,
        items: Optional[List[Slice]] = None,
        volume_meta: Optional[dict] = None,
    ):
        Plane.validate_name(plane_name)
        self._name = plane_name

        if img_size is None and volume_meta is None:
            raise ValueError(
                "Both img_size and volume_meta are None, only one of them is allowed to be a None"
            )
        if slices_count is None and volume_meta is None:
            raise ValueError(
                "Both slices_count and volume_meta are None, only one of them is allowed to be a None"
            )
        self._img_size = take_with_default(img_size, Plane.get_img_size(self._name, volume_meta))
        self._img_size = validate_img_size(self._img_size)

        self._slices_count = take_with_default(
            slices_count, Plane.get_slices_count(self._name, volume_meta)
        )

        super().__init__(items=items)

    @property
    def name(self) -> Union[str, None]:
        """
        Get the name of the plane.

        :return: Name of the plane.
        :rtype: Union[str, None]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/Users/almaz/Downloads/my volumes/ds11111/Demo volumes_ds1_CTChest.nrrd"
            vol, meta = sly.volume.read_nrrd_serie_volume(path)

            plane = sly.Plane(
                sly.Plane.AXIAL,
                volume_meta=meta,
            )
            print(plane.name)
            # Output: axial
        """

        return self._name

    @property
    def slices_count(self) -> int:
        """
        Get the number of slices in the plane.

        :return: Number of slices in the plane.
        :rtype: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/Users/almaz/Downloads/my volumes/ds11111/Demo volumes_ds1_CTChest.nrrd"
            vol, meta = sly.volume.read_nrrd_serie_volume(path)

            plane = sly.Plane(
                sly.Plane.AXIAL,
                volume_meta=meta,
            )
            print(plane.slices_count)
            # Output: 139
        """

        return self._slices_count

    @property
    def img_size(self) -> Tuple[int]:
        """
        Get the size of the image in the plane.

        :return: Size of the image in the plane.
        :rtype: Tuple[int]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/Users/almaz/Downloads/my volumes/ds11111/Demo volumes_ds1_CTChest.nrrd"
            vol, meta = sly.volume.read_nrrd_serie_volume(path)

            plane = sly.Plane(
                sly.Plane.AXIAL,
                volume_meta=meta,
            )
            print(plane.img_size)
            # Output: (512, 512)
        """

        return deepcopy(self._img_size)

    @property
    def normal(self) -> dict:
        """
        Returns the normal vector of the plane.

        :return: A dictionary representing the normal vector of the plane.
        :rtype: dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/Users/almaz/Downloads/my volumes/ds11111/Demo volumes_ds1_CTChest.nrrd"
            vol, meta = sly.volume.read_nrrd_serie_volume(path)

            plane = sly.Plane(
                sly.Plane.AXIAL,
                volume_meta=meta,
            )
            print(plane.normal)
            # Output: {'x': 0, 'y': 0, 'z': 1}
        """

        return Plane.get_normal(self.name)

    def __str__(self):
        return super().__str__().replace("Frames", "Slices")

    @staticmethod
    def get_normal(name: str) -> dict:
        """
        Returns the normal vector of a plane given its name.

        :param name: Name of the plane.
        :type name: str
        :return: A dictionary representing the normal vector of the plane.
        :rtype: dict
        :raises ValueError: If `name` is not one of "sagittal", "coronal", or "axial".
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            print(sly.Plane.get_normal(sly.Plane.AXIAL))
            # Output: {'x': 0, 'y': 0, 'z': 1}
        """

        Plane.validate_name(name)
        if name == Plane.SAGITTAL:
            return {"x": 1, "y": 0, "z": 0}
        if name == Plane.CORONAL:
            return {"x": 0, "y": 1, "z": 0}
        if name == Plane.AXIAL:
            return {"x": 0, "y": 0, "z": 1}

    @staticmethod
    def get_name(normal: dict) -> str:
        """
        Returns the name of a plane given its normal vector.

        :param normal: A dictionary representing the normal vector of a plane.
        :type normal: dict
        :return: The name of the plane.
        :rtype: str
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            print(sly.Plane.get_name({'x': 0, 'y': 0, 'z': 1}))
            # Output: axial
        """

        if normal == {"x": 1, "y": 0, "z": 0}:
            return Plane.SAGITTAL
        if normal == {"x": 0, "y": 1, "z": 0}:
            return Plane.CORONAL
        if normal == {"x": 0, "y": 0, "z": 1}:
            return Plane.AXIAL

    @staticmethod
    def get_img_size(name: str, volume_meta: dict) -> List[int]:
        """
        Get size of the image for a given plane and volume metadata.

        :param name: Name of the plane.
        :type name: str
        :param volume_meta: Metadata for the volume.
        :type volume_meta: dict
        :return: The size of the image for the given plane.
        :rtype: List[int]
        :raises ValueError: If `name` is not one of "sagittal", "coronal", or "axial".
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/Users/almaz/Downloads/my volumes/ds11111/Demo volumes_ds1_CTChest.nrrd"
            vol, meta = sly.volume.read_nrrd_serie_volume(path)

            plane = sly.Plane(
                sly.Plane.AXIAL,
                volume_meta=meta,
            )
            img_size = sly.Plane.get_img_size(plane.name, meta)
            print(img_size)
            # Output: [512, 512]
        """

        Plane.validate_name(name)
        dimentions = volume_meta["dimensionsIJK"]
        # (height, width)
        height = None
        width = None
        if name == Plane.SAGITTAL:
            width = dimentions["y"]
            height = dimentions["z"]
        elif name == Plane.CORONAL:
            width = dimentions["x"]
            height = dimentions["z"]
        elif name == Plane.AXIAL:
            width = dimentions["x"]
            height = dimentions["y"]
        return [height, width]

    @staticmethod
    def get_slices_count(name: str, volume_meta: dict) -> int:
        """
        Returns the number of slices in the given plane.

        :param name: Name of the plane.
        :type name: str
        :param volume_meta: Metadata of the volume to extract slices from.
        :type volume_meta: dict
        :return: Number of slices in the given plane.
        :rtype: int
        :raises ValueError: If `name` is not one of "sagittal", "coronal", or "axial".
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/Users/almaz/Downloads/my volumes/ds11111/Demo volumes_ds1_CTChest.nrrd"
            vol, meta = sly.volume.read_nrrd_serie_volume(path)

            plane = sly.Plane(
                sly.Plane.AXIAL,
                volume_meta=meta,
            )
            slices_count = sly.Plane.get_slices_count(plane.name, meta)
            print(slices_count)
            # Output: 139
        """

        Plane.validate_name(name)
        dimentions = volume_meta["dimensionsIJK"]
        if name == Plane.SAGITTAL:
            return dimentions["x"]
        elif name == Plane.CORONAL:
            return dimentions["y"]
        elif name == Plane.AXIAL:
            return dimentions["z"]

    @classmethod
    def from_json(
        cls,
        data: dict,
        plane_name: str,
        objects: VolumeObjectCollection,
        img_size: Optional[Union[list, tuple]] = None,
        slices_count: Optional[int] = None,
        volume_meta: Optional[dict] = None,
        key_id_map: Optional[KeyIdMap] = None,
    ):
        """
        Creates a `Plane` instance from a JSON dictionary.

        :param data: JSON dictionary representing a `Plane` instance.
        :type data: dict
        :param plane_name: Name of the plane.
        :type plane_name: str
        :param objects: Objects in the plane.
        :type objects: VolumeObjectCollection
        :param img_size: Size of the image represented by the plane.
        :type img_size: Optional[Union[list, tuple]]
        :param slices_count: Number of slices along the plane.
        :type slices_count: Optional[int]
        :param volume_meta: Metadata of the volume to extract slices from.
        :type volume_meta: Optional[dict]
        :param key_id_map: Dictionary mapping object keys to object IDs.
        :type key_id_map: Optional[KeyIdMap]
        :return: A new class:`Plane<Plane>` instance created from the JSON.
        :rtype: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`

        :raises ValueError: If `plane_name` is not equal to the "name" field in "data", or if the "normal" field in "data" is not valid for the given plane, or if both `slices_count` and `volume_meta` are None.
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Create Plane from json we use data from example to_json(see below)
            path = "/Users/almaz/Downloads/my volumes/ds11111/Demo volumes_ds1_CTChest.nrrd"
            new_plane = sly.Plane.from_json(
                data=plane.to_json(),
                plane_name=sly.Plane.AXIAL,
                objects=sly.VolumeObjectCollection([]),
                volume_meta=meta,
            )
        """

        Plane.validate_name(plane_name)
        if plane_name != data[constants.NAME]:
            raise ValueError(
                f"Plane name {plane_name} differs from the name in json data {data[constants.NAME]}"
            )

        if Plane.get_normal(plane_name) != data[constants.NORMAL]:
            raise ValueError(
                f"Normal json data {data[constants.NORMAL]} is not valid for {plane_name}. It should be {Plane.get_normal(plane_name)}"
            )
        if slices_count is None:
            if volume_meta is None:
                raise ValueError(
                    "Both slices_count and volume_meta are None, only one of them is allowed to be a None"
                )
            else:
                slices_count = Plane.get_slices_count(plane_name, volume_meta)

        slices = []
        for slice_json in data[constants.SLICES]:
            slices.append(
                cls.item_type.from_json(slice_json, objects, plane_name, slices_count, key_id_map)
            )

        return cls(plane_name, img_size, slices_count, slices, volume_meta)

    def to_json(
        self,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> dict:
        """
        Returns a JSON serializable dictionary representation of the `Plane` instance.

        :param key_id_map: Dictionary mapping object keys to object IDs.
        :type key_id_map: Optional[KeyIdMap]
        :return: A JSON serializable dictionary representation of the `Plane` instance.
        :rtype: dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.volume_annotation.plane import Plane
            path = "/Users/almaz/Downloads/my volumes/ds11111/Demo volumes_ds1_CTChest.nrrd"
            vol, meta = sly.volume.read_nrrd_serie_volume(path)

            plane = Plane(
                sly.Plane.AXIAL,
                volume_meta=meta,
            )
            print(plane.to_json())
            # Output: { "name": "axial", "normal": { "x": 0, "y": 0, "z": 1 }, "slices": [] }
        """
        json_slices = []
        for slice in self:
            slice: Slice
            json_slices.append(slice.to_json(key_id_map))

        return {
            constants.NAME: self.name,
            constants.NORMAL: self.normal,
            constants.SLICES: json_slices,
        }

    def validate_figures_bounds(self):
        """
        Validates the figure bounds for all slices in the Plane.

        :raises ValueError: If any of the slices in the `Plane` instance have invalid figure bounds.
        """
        for slice in self:
            slice: Slice
            slice.validate_figures_bounds(self.img_size)
