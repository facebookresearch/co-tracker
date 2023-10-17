# coding: utf-8

# docs
from __future__ import annotations
import numpy as np
import base64
import gzip
import nrrd
import tempfile
from typing import Optional, Union, List, Tuple, Dict, Literal
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import (
    SPACE_ORIGIN,
    DATA,
    GEOMETRY_SHAPE,
    GEOMETRY_TYPE,
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    ID,
    CLASS_ID,
    MASK_3D,
)
from supervisely._utils import unwrap_if_numpy
from supervisely.io.json import JsonSerializable
from supervisely.io.fs import remove_dir
from supervisely import logger


if not hasattr(np, "bool"):
    np.bool = np.bool_


class PointVolume(JsonSerializable):
    """
    PointVolume (x, y, z) determines position of Mask3D. It locates the first sample.
    :class:`PointVolume<PointVolume>` object is immutable.

    :param x: Position of PointVolume object on X-axis.
    :type x: int or float
    :param y: Position of PointVolume object on Y-axis.
    :type y: int or float
    :param z: Position of PointVolume object on Z-axis.
    :type z: int or float
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        x = 100
        y = 200
        z = 2
        loc = sly.PointVolume(x, y, z)
    """

    def __init__(self, x: Union[int, float], y: Union[int, float], z: Union[int, float]):
        self._x = round(unwrap_if_numpy(x))
        self._y = round(unwrap_if_numpy(y))
        self._z = round(unwrap_if_numpy(z))

    @property
    def x(self) -> int:
        """
        Position of PointVolume on X-axis.

        :return: X of PointVolume
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(loc.x)
            # Output: 100
        """
        return self._x

    @property
    def y(self) -> int:
        """
        Position of PointVolume on Y-axis.

        :return: Y of PointVolume
        :rtype: :class:`int`

        :Usage example:

         .. code-block:: python

            print(loc.y)
            # Output: 200
        """
        return self._y

    @property
    def z(self) -> int:
        """
        Position of PointVolume on Z-axis.

        :return: Z of PointVolume
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(loc.z)
            # Output: 2
        """
        return self._z

    def to_json(self) -> Dict:
        """
        Convert the PointVolume to a json dict.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            loc_json = loc.to_json()
            print(loc_json)
            # Output: {
            #           "space_origin": [
            #                            200,
            #                            200,
            #                            100
            #                           ]
            #         }

        """

        packed_obj = {SPACE_ORIGIN: [self.x, self.y, self.z]}
        return packed_obj

    @classmethod
    def from_json(cls, packed_obj) -> PointVolume:
        """
        Convert a json dict to PointVolume.

        :param data: PointVolume in json format as a dict.
        :type data: dict
        :return: PointVolume object
        :rtype: :class:`PointVolume<PointVolume>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            loc_json = {
                "space_origin": [
                                200,
                                200,
                                100,
                                ]
                        }

            loc = sly.PointVolume.from_json(loc_json)
        """
        return cls(
            x=packed_obj["space_origin"][0],
            y=packed_obj["space_origin"][1],
            z=packed_obj["space_origin"][2],
        )


class Mask3D(Geometry):
    """
    Mask 3D geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Mask3D<Mask3D>` object is immutable.

    :param data: Mask 3D mask data. Must be a numpy array with only 2 unique values: [0, 1] or [0, 255] or [False, True].
    :type data: np.ndarray
    :param sly_id: Mask 3D ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Mask 3D belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Mask 3D.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Mask 3D was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Mask 3D was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`, if data is not bool or no pixels set to True in data
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create simple Mask 3D
        mask3d = np.zeros((3, 3, 3), dtype=np.bool_)
        mask3d[0:2, 0:2, 0:2] = True

        shape = sly.Mask3D(mask3d)

        print(shape.data)
        # Output:
        #    [[[ True  True False]
        #      [ True  True False]
        #      [False False False]]

        #     [[ True  True False]
        #      [ True  True False]
        #      [False False False]]

        #     [[False False False]
        #      [False False False]
        #      [False False False]]]
    """

    def __init__(
        self,
        data: np.ndarray,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

        if not isinstance(data, np.ndarray):
            raise TypeError('Mask 3D "data" argument must be numpy array object!')

        data_dims = len(data.shape)
        if data_dims != 3:
            raise ValueError(
                f'Mask 3D "data" argument must be a 3-dimensional numpy array. Instead got {data_dims} dimensions'
            )

        if data.dtype != np.bool:
            if list(np.unique(data)) not in [[0, 1], [0, 255]]:
                raise ValueError(
                    f"Mask 3D mask data values must be one of: [0 1], [0 255], [False True]. Instead got {np.unique(data)}."
                )

            if list(np.unique(data)) == [0, 1]:
                data = np.array(data, dtype=bool)
            elif list(np.unique(data)) == [0, 255]:
                data = np.array(data / 255, dtype=bool)

        self.data = data
        self._space_origin = None
        self._space = None
        self._space_directions = None

    @staticmethod
    def geometry_name():
        """Return geometry name"""
        return "mask_3d"

    @staticmethod
    def from_file(figure, file_path: str):
        """
        Load figure geometry from file.

        :param figure: Spatial figure
        :type figure: VolumeFigure
        :param file_path: Path to nrrd file with data
        :type file_path: str
        """
        mask3d_data, mask3d_header = nrrd.read(file_path)
        figure.geometry.data = mask3d_data
        figure.geometry._space_origin = PointVolume(
            x=mask3d_header["space origin"][0],
            y=mask3d_header["space origin"][1],
            z=mask3d_header["space origin"][2],
        )
        figure.geometry._space = mask3d_header["space"]
        figure.geometry._space_directions = mask3d_header["space directions"]
        path_without_filename = "/".join(file_path.split("/")[:-1])
        remove_dir(path_without_filename)

    @classmethod
    def create_from_file(cls, file_path: str) -> Mask3D:
        """
        Creates Mask3D geometry from file.

        :param file_path: Path to nrrd file with data
        :type file_path: str
        """
        mask3d_data, mask3d_header = nrrd.read(file_path)
        geometry = cls(data=mask3d_data)
        try:
            geometry._space_origin = PointVolume(
                x=mask3d_header["space origin"][0],
                y=mask3d_header["space origin"][1],
                z=mask3d_header["space origin"][2],
            )
            geometry._space = mask3d_header["space"]
            geometry._space_directions = mask3d_header["space directions"]
        except KeyError:
            logger.debug(
                "The Mask3D geometry created from the file does not contain private attributes"
            )
        return geometry

    @classmethod
    def from_bytes(cls, geometry_bytes: bytes) -> Mask3D:
        """
        Create a Mask3D geometry object from bytes.

        :param geometry_bytes: NRRD file represented as bytes.
        :type geometry_bytes: bytes
        :return: A Mask3D geometry object.
        :rtype: Mask3D
        """
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(geometry_bytes)
            data_array, _ = nrrd.read(temp_file.name)

        return cls(data_array)

    def to_json(self) -> Dict:
        """
        Convert the Mask 3D to a json dict.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            mask = np.array([[[1 1 0]
                              [1 1 0]
                              [0 0 0]]
                             [[1 1 0]
                              [1 1 0]
                              [0 0 0]]
                             [[0 0 0]
                              [0 0 0]
                              [0 0 0]]], dtype=np.bool_)

            figure = sly.Mask3D(mask)
            figure_json = figure.to_json()
            print(json.dumps(figure_json, indent=4))
            # Output: {
            #    "mask_3d": {
            #        "data": "eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6",
            #    },
            #    "shape": "mask_3d",
            #    "geometryType": "mask_3d"
            # }
        """
        res = {
            self._impl_json_class_name(): {
                DATA: self.data_2_base64(self.data),
            },
            GEOMETRY_SHAPE: self.name(),
            GEOMETRY_TYPE: self.name(),
        }

        if self._space_origin:
            res[f"{self._impl_json_class_name()}"][f"{SPACE_ORIGIN}"] = [
                self._space_origin.x,
                self._space_origin.y,
                self._space_origin.z,
            ]

        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, json_data: Dict) -> Mask3D:
        """
        Convert a json dict to Mask 3D.

        :param data: Mask in json format as a dict.
        :type data: dict
        :return: Mask3D object
        :rtype: :class:`Mask3D<Mask3D>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_json = {
                "mask_3d": {
                    "data": "eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6",
                },
                "shape": "mask_3d",
                "geometryType": "mask_3d"
            }

            figure = sly.Mask3D.from_json(figure_json)
        """
        if json_data == {}:
            return cls(data=np.zeros((3, 3, 3), dtype=np.bool_))

        json_root_key = cls._impl_json_class_name()
        if json_root_key not in json_data:
            raise ValueError(
                "Data must contain {} field to create Mask3D object.".format(json_root_key)
            )

        if DATA not in json_data[json_root_key]:
            raise ValueError(
                "{} field must contain {} and {} fields to create Mask3D object.".format(
                    json_root_key, DATA
                )
            )

        data = cls.base64_2_data(json_data[json_root_key][DATA])

        labeler_login = json_data.get(LABELER_LOGIN, None)
        updated_at = json_data.get(UPDATED_AT, None)
        created_at = json_data.get(CREATED_AT, None)
        sly_id = json_data.get(ID, None)
        class_id = json_data.get(CLASS_ID, None)
        instance = cls(
            data=data.astype(np.bool_),
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        if SPACE_ORIGIN in json_data[json_root_key]:
            x, y, z = json_data[json_root_key][SPACE_ORIGIN]
            instance._space_origin = PointVolume(x=x, y=y, z=z)
        return instance

    @classmethod
    def _impl_json_class_name(cls):
        """_impl_json_class_name"""
        return MASK_3D

    @staticmethod
    def data_2_base64(data: np.ndarray) -> str:
        """
        Convert numpy array to base64 encoded string.

        :param mask: Bool numpy array.
        :type mask: np.ndarray
        :return: Base64 encoded string
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import os
            import nrrd

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            meta_json = api.project.get_meta(PROJECT_ID)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_json = api.volume.annotation.download_bulk(DATASET_ID, [VOLUME_ID])

            figure_id = ann_json[0]["spatialFigures"][0]["id"]
            path_for_mesh = f"meshes/{figure_id}.nrrd"
            api.volume.figure.download_stl_meshes([figure_id], [path_for_mesh])

            mask3d_data, _ = nrrd.read(path_for_mesh)
            encoded_string = sly.Mask3D.data_2_base64(mask3d_data)

            print(encoded_string)
            # 'H4sIAGWoWmQC/zPWMdYxrmFkZAAiIIAz4AAAE56ciyEAAAA='
        """
        shape_str = ",".join(str(dim) for dim in data.shape)
        data_str = data.tostring().decode("utf-8")
        combined_str = f"{shape_str}|{data_str}"
        compressed_string = gzip.compress(combined_str.encode("utf-8"))
        encoded_string = base64.b64encode(compressed_string).decode("utf-8")
        return encoded_string

    @staticmethod
    def base64_2_data(encoded_string: str) -> np.ndarray:
        """
        Convert base64 encoded string to numpy array.

        :param s: Input base64 encoded string.
        :type s: str
        :return: Bool numpy array
        :rtype: :class:`np.ndarray`
        :Usage example:

         .. code-block:: python

              import supervisely as sly

              encoded_string = 'H4sIAGWoWmQC/zPWMdYxrmFkZAAiIIAz4AAAE56ciyEAAAA='
              figure_data = sly.Mask3D.base64_2_data(encoded_string)
              print(figure_data)
              # [[[1 1 0]
              #   [1 1 0]
              #   [0 0 0]]
              #  [[1 1 0]
              #   [1 1 0]
              #   [0 0 0]]
              #  [[0 0 0]
              #   [0 0 0]
              #   [0 0 0]]]
        """
        compressed_bytes = base64.b64decode(encoded_string)
        decompressed_string = gzip.decompress(compressed_bytes).decode("utf-8")
        shape_str, data_str = decompressed_string.split("|")
        shape = tuple(int(dim) for dim in shape_str.split(","))
        data_bytes = data_str.encode("utf-8")
        try:
            data = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)
        except ValueError:
            logger.warn(
                "Can't reshape array with 'dtype=np.uint8'. Will try to automatically convert 'dtype=np.int16' to 'np.uint8' and reshape"
            )
            data = np.frombuffer(data_bytes, dtype=np.int16)
            data = np.clip(data, 0, 1).astype(np.uint8)
            data = data.reshape(shape)
            logger.debug("Converted successfully!")
        return data

    def add_mask_2d(
        self,
        mask_2d: np.ndarray,
        plane_name: Literal["axial", "sagittal", "coronal"],
        slice_index: int,
        origin: Optional[List[int]] = None,
    ):
        """
        Draw a 2D mask on a 3D Mask.

        :param mask_2d: 2D array with a flat mask.
        :type mask_2d: np.ndarray
        :param plane_name: Name of the plane: "axial", "sagittal", "coronal".
        :type plane_name: str
        :param slice_index: Slice index of the volume figure.
        :type slice_index: int
        :param origin: (row, col) position. The top-left corner of the mask is located on the specified slice (optional).
        :type origin: Optional[List[int]], NoneType
        """

        from supervisely.volume_annotation.plane import Plane

        Plane.validate_name(plane_name)

        mask_2d = np.fliplr(mask_2d)
        mask_2d = np.rot90(mask_2d, 1, (1, 0))

        if plane_name == Plane.AXIAL:
            new_shape = self.data.shape[:2]
        elif plane_name == Plane.SAGITTAL:
            new_shape = self.data.shape[1:]
        elif plane_name == Plane.CORONAL:
            new_shape = self.data.shape[::2]

        if origin:
            x, y = origin
            new_mask = np.zeros(new_shape, dtype=mask_2d.dtype)
            new_mask[x : x + mask_2d.shape[0], y : y + mask_2d.shape[1]] = mask_2d

        if plane_name == Plane.AXIAL:
            self.data[:, :, slice_index] = new_mask
        elif plane_name == Plane.SAGITTAL:
            self.data[slice_index, :, :] = new_mask
        elif plane_name == Plane.CORONAL:
            self.data[:, slice_index, :] = new_mask
