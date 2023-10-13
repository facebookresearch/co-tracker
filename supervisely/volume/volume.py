# coding: utf-8
"""Functions for processing volumes"""

import os
import json
from typing import List, Tuple, Union
import numpy as np

import pydicom
import SimpleITK as sitk
import stringcase
from supervisely.io.fs import get_file_ext, list_files_recursively, list_files
import supervisely.volume.nrrd_encoder as nrrd_encoder
from supervisely import logger

# Do NOT use directly for extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_VOLUME_EXTENSIONS = [".nrrd", ".dcm"]


class UnsupportedVolumeFormat(Exception):
    pass


def get_extension(path: str):
    """
    Get extension for given path.

    :param path: Path to volume.
    :type path: str
    :return: Path extension
    :rtype: str
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "src/upload/folder/CTACardio.nrrd"
        ext = sly.volume.get_extension(path=path) # .nrrd
    """

    # magic.from_file("path", mime=True)
    # for nrrd:
    # application/octet-stream
    # for nifti(nii):
    # application/octet-stream
    # for dicom:
    # "application/dicom"

    ext = get_file_ext(path)
    if ext in ALLOWED_VOLUME_EXTENSIONS:
        return ext

    # case when dicom file does not have an extension
    import magic

    mime = magic.from_file(path, mime=True)
    if mime == "application/dicom":
        return ".dcm"
    return None


def is_valid_ext(ext: str) -> bool:
    """
    Checks if given extension is supported.

    :param ext: Volume file extension.
    :type ext: str
    :return: True if extensions is in the list of supported extensions else False
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        sly.volume.is_valid_ext(".nrrd")  # True
        sly.volume.is_valid_ext(".mp4") # False
    """

    return ext.lower() in ALLOWED_VOLUME_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    """
    Checks if Volume file from given path has supported extension.

    :param path: Path to volume file.
    :type path: str
    :return: True if Volume file has supported extension else False
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        volume_path = "/home/admin/work/volumes/vol_01.nrrd"
        sly.volume.has_valid_ext(volume_path) # True
    """

    return is_valid_ext(get_extension(path))


def validate_format(path: str):
    """
    Raise error if Volume file from given path couldn't be read or file extension is not supported.

    :param path: Path to Volume file.
    :type path: str
    :raises: :class:`UnsupportedVolumeFormat` if Volume file from given path couldn't be read or file extension is not supported.
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        volume_path = "/home/admin/work/volumes/vol_01.mp4"
        sly.volume.validate_format(volume_path)
        # File /home/admin/work/volumes/vol_01.mp4 has unsupported volume extension. Supported extensions: [".nrrd", ".dcm"].
    """

    if not has_valid_ext(path):
        raise UnsupportedVolumeFormat(
            f"File {path} has unsupported volume extension. Supported extensions: {ALLOWED_VOLUME_EXTENSIONS}"
        )


def rescale_slope_intercept(value: float, slope: float, intercept: float) -> float:
    """
    Rescale intensity value using the given slope and intercept.

    :param value: The intensity value to be rescaled.
    :type value: float
    :param slope: The slope for rescaling.
    :type slope: float
    :param intercept: The intercept for rescaling.
    :type intercept: float
    :return: The rescaled intensity value.
    :rtype: float

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        meta["intensity"]["min"] = sly.volume.volume.rescale_slope_intercept(
            meta["intensity"]["min"],
            meta["rescaleSlope"],
            meta["rescaleIntercept"],
        )
    """

    return value * slope + intercept


def normalize_volume_meta(meta: dict) -> dict:
    """
    Normalize volume metadata.

    :param meta: Metadata of the volume.
    :type meta: dict
    :return: Normalized volume metadata.
    :rtype: dict

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        normalized_meta = sly.volume.volume.volume.normalize_volume_meta(volume_meta)

        print(normalized_meta)
        # Output:
        # {
        #     'ACS': 'RAS',
        #     'channelsCount': 1,
        #     'dimensionsIJK': {'x': 512, 'y': 512, 'z': 139},
        #     'directions': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        #     'intensity': {'max': 3071.0, 'min': -3024.0},
        #     'origin': (-194.238403081894, -217.5384061336518, -347.7500000000001),
        #     'rescaleIntercept': 0,
        #     'rescaleSlope': 1,
        #     'spacing': (0.7617189884185793, 0.7617189884185793, 2.5),
        #     'windowCenter': 23.5,
        #     'windowWidth': 6095.0
        # }
    """

    meta["intensity"]["min"] = rescale_slope_intercept(
        meta["intensity"]["min"],
        meta["rescaleSlope"],
        meta["rescaleIntercept"],
    )

    meta["intensity"]["max"] = rescale_slope_intercept(
        meta["intensity"]["max"],
        meta["rescaleSlope"],
        meta["rescaleIntercept"],
    )

    if "windowWidth" not in meta:
        meta["windowWidth"] = meta["intensity"]["max"] - meta["intensity"]["min"]

    if "windowCenter" not in meta:
        meta["windowCenter"] = meta["intensity"]["min"] + meta["windowWidth"] / 2

    return meta


def read_dicom_serie_volume_np(paths: List[str], anonymize=True) -> Tuple[np.ndarray, dict]:
    """
    Read DICOM series volumes with given paths.

    :param paths: Paths to DICOM volume files.
    :type paths: List[str]
    :param anonymize: Specify whether to hide PatientID and PatientName fields.
    :type anonymize: bool
    :return: Volume data in NumPy array format and dictionary with metadata
    :rtype: Tuple[np.ndarray, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        volume_path = ["/home/admin/work/volumes/vol_01.nrrd"]
        volume_np, meta = sly.volume.read_dicom_serie_volume_np(volume_path)
    """

    import SimpleITK as sitk

    sitk_volume, meta = read_dicom_serie_volume(paths, anonymize=anonymize)
    # for debug:
    # sitk.WriteImage(sitk_volume, "/work/output/sitk.nrrd", useCompression=False, compressionLevel=9)
    # with open("/work/output/test.nrrd", "wb") as file:
    #     file.write(b)
    volume_np = sitk.GetArrayFromImage(sitk_volume)
    volume_np = np.transpose(volume_np, (2, 1, 0))
    return volume_np, meta


_anonymize_tags = ["PatientID", "PatientName"]
_default_dicom_tags = [
    "SeriesInstanceUID",
    "Modality",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
    "PhotometricInterpretation",
]
_default_dicom_tags.extend(_anonymize_tags)

_photometricInterpretationRGB = set(
    [
        "RGB",
        "PALETTE COLOR",
        "YBR_FULL",
        "YBR_FULL_422",
        "YBR_PARTIAL_422",
        "YBR_PARTIAL_420",
        "YBR_RCT",
    ]
)


def read_dicom_tags(
    path: str,
    allowed_keys: Union[None, List[str]] = _default_dicom_tags,
    anonymize: bool = True,
):
    """
    Read DICOM tags from a DICOM file.
    
    :param path: Path to the DICOM file.
    :type path: str
    :param allowed_keys: List of allowed DICOM keywords to be extracted. Default is None, which means all keywords are allowed.
    :type allowed_keys: Union[None, List[str]], optional
    :param anonymize: Flag to indicate whether to anonymize certain tags or not.
    :type anonymize: bool, optional
    :return: Dictionary containing the extracted DICOM tags.
    :rtype: dict

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "src/upload/Dicom_files/nnn.dcm"
        dicom_tags = sly.volume.read_dicom_tags(path=path)
    """

    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    vol_info = {}
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        tag = pydicom.tag.Tag(k.split("|")[0], k.split("|")[1])
        keyword = pydicom.datadict.keyword_for_tag(tag)
        if allowed_keys is not None and keyword not in allowed_keys:
            continue
        if anonymize is True and keyword in _anonymize_tags:
            v = "anonymized"
        keyword = stringcase.camelcase(keyword)
        vol_info[keyword] = v
        if keyword in [
            "windowCenter",
            "windowWidth",
            "rescaleIntercept",
            "rescaleSlope",
        ]:
            vol_info[keyword] = float(vol_info[keyword].split("\\")[0])
        elif keyword == "photometricInterpretation" and v in _photometricInterpretationRGB:
            vol_info["channelsCount"] = 3
    return vol_info


def encode(volume_np: np.ndarray, volume_meta: dict) -> bytes:
    """
    Encodes a volume from NumPy format into a NRRD format.

    :param volume_np: NumPy array representing the volume data.
    :type volume_np: np.ndarray
    :param volume_meta: Metadata of the volume.
    :type volume_meta: dict

    :return: Encoded volume data in bytes.
    :rtype: bytes

    :Usage example:

     .. code-block:: python

        import numpy as np
        import supervisely as sly

        volume_np = np.random.rand(256, 256, 256)
        volume_meta = {
            "ACS": "RAS",
            "channelsCount": 1,
            "dimensionsIJK": { "x": 512, "y": 512, "z": 139 },
            "directions": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            "intensity": { "max": 3071.0, "min": -3024.0 },
            "origin": (-194.238403081894, -217.5384061336518, -347.7500000000001),
            "rescaleIntercept": 0,
            "rescaleSlope": 1,
            "spacing": (0.7617189884185793, 0.7617189884185793, 2.5),
            "windowCenter": 23.5,
            "windowWidth": 6095.0
        }

        encoded_volume = sly.volume.encode(volume_np, volume_meta)
    """

    directions = np.array(volume_meta["directions"]).reshape(3, 3)
    directions *= volume_meta["spacing"]

    volume_bytes = nrrd_encoder.encode(
        volume_np,
        header={
            "encoding": "gzip",
            # "space": "left-posterior-superior",
            "space": "right-anterior-superior",
            "space directions": directions.T.tolist(),
            "space origin": volume_meta["origin"],
        },
        compression_level=1,
    )

    # with open("/work/output/test.nrrd", "wb") as file:
    #     file.write(volume_bytes)

    return volume_bytes


def inspect_dicom_series(root_dir: str):
    """
    Search for DICOM series in the directory and its subdirectories.

    :param root_dir: Directory path with volumes.
    :type root_dir: str
    :return: Dictionary with DICOM volumes IDs and corresponding fiel names.
    :rtype: dict
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "src/upload/Dicom_files/"
        series_infos = sly.volume.inspect_dicom_series(root_dir=path)
    """
    import SimpleITK as sitk

    found_series = {}
    for d in os.walk(root_dir):
        dir = d[0]
        reader = sitk.ImageSeriesReader()
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        series_found = reader.GetGDCMSeriesIDs(dir)
        sitk.ProcessObject_SetGlobalWarningDisplay(True)
        logger.info(f"Found {len(series_found)} series in directory {dir}")
        for serie in series_found:
            dicom_names = reader.GetGDCMSeriesFileNames(dir, serie)
            found_series[serie] = dicom_names
    logger.info(f"Total {len(found_series)} series in directory {root_dir}")
    return found_series


def _sitk_image_orient_ras(sitk_volume):
    import SimpleITK as sitk

    if sitk_volume.GetDimension() == 4 and sitk_volume.GetSize()[3] == 1:
        sitk_volume = sitk_volume[:, :, :, 0]

    sitk_volume = sitk.DICOMOrient(sitk_volume, "RAS")
    # RAS reorient image using filter
    # orientation_filter = sitk.DICOMOrientImageFilter()
    # orientation_filter.SetDesiredCoordinateOrientation("RAS")
    # sitk_volume = orientation_filter.Execute(sitk_volume)

    # https://discourse.itk.org/t/getdirection-and-getorigin-for-simpleitk-c-implementation/3472/8
    origin = list(sitk_volume.GetOrigin())
    directions = list(sitk_volume.GetDirection())
    origin[0] *= -1
    origin[1] *= -1
    directions[0] *= -1
    directions[1] *= -1
    directions[3] *= -1
    directions[4] *= -1
    directions[6] *= -1
    directions[7] *= -1
    sitk_volume.SetOrigin(origin)
    sitk_volume.SetDirection(directions)
    return sitk_volume


def read_dicom_serie_volume(paths: List[str], anonymize: bool = True) -> Tuple[sitk.Image, dict]:
    """
    Read DICOM series volumes with given paths.

    :param paths: Paths to DICOM volume files.
    :type paths: List[str]
    :param anonymize: Specify whether to hide PatientID and PatientName fields.
    :type anonymize: bool
    :return: Volume data in SimpleITK.Image format and dictionary with metadata.
    :rtype: Tuple[SimpleITK.Image, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        paths = ["/home/admin/work/volumes/vol_01.nrrd"]
        sitk_volume, meta = sly.volume.read_dicom_serie_volume(paths)
    """

    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    sitk_volume = reader.Execute()
    sitk_volume = _sitk_image_orient_ras(sitk_volume)
    dicom_tags = read_dicom_tags(paths[0], anonymize=anonymize)

    f_min_max = sitk.MinimumMaximumImageFilter()
    f_min_max.Execute(sitk_volume)
    meta = get_meta(
        sitk_volume.GetSize(),
        f_min_max.GetMinimum(),
        f_min_max.GetMaximum(),
        sitk_volume.GetSpacing(),
        sitk_volume.GetOrigin(),
        sitk_volume.GetDirection(),
        dicom_tags,
    )
    return sitk_volume, meta


def compose_ijk_2_world_mat(meta: dict) -> np.ndarray:
    """
    Transform 4x4 matrix from voxels to world coordinates.

    :param meta: Volume metadata.
    :type meta: dict
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        mat = sly.volume.volume.compose_ijk_2_world_mat(volume_meta)

        # Output:
        # [
        #     [   0.76171899    0.            0.         -194.23840308]
        #     [   0.            0.76171899    0.         -217.53840613]
        #     [   0.            0.            2.5        -347.75      ]
        #     [   0.            0.            0.            1.        ]
        # ]
    """

    try:
        spacing = meta["spacing"]
        origin = meta["origin"]
        directions = meta["directions"]
    except KeyError as e:
        raise IOError(
            f"Need the meta '{e}'' field to determine the mapping from voxels to world coordinates."
        )

    mat = np.eye(4)
    mat[:3, :3] = (np.array(directions).reshape(3, 3) * spacing).T
    mat[:3, 3] = origin
    return mat


def world_2_ijk_mat(ijk_2_world) -> np.ndarray:
    """
    Transform 4x4 matrix from world to voxels coordinates.

    :param ijk_2_world: 4x4 matrix.
    :type ijk_2_world: np.ndarray
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        mat = sly.volume.volume.world_2_ijk_mat(world_mat)

        # Output:
        # [
        #     [  1.3128201    0.           0.         255.00008013]
        #     [  0.           1.3128201    0.         285.58879251]
        #     [  0.           0.           0.4        139.1       ]
        #     [  0.           0.           0.           1.        ]
        # ]
    """

    return np.linalg.inv(ijk_2_world)


def get_meta(
    sitk_shape: tuple,
    min_intensity: float,
    max_intensity: float,
    spacing: tuple,
    origin: tuple,
    directions: tuple,
    dicom_tags: dict = {},
) -> dict:
    """
    Get normalized meta-data for a volume.

    :param sitk_shape: Tuple representing the shape of the volume in (x, y, z) dimensions.
    :type sitk_shape: tuple
    :param min_intensity: Minimum intensity value in the volume.
    :type min_intensity: float
    :param max_intensity: Maximum intensity value in the volume.
    :type max_intensity: float
    :param spacing: Tuple representing the spacing between voxels in (x, y, z) dimensions.
    :type spacing: tuple
    :param origin: Tuple representing the origin of the volume in (x, y, z) dimensions.
    :type origin: tuple
    :param directions: Tuple representing the direction matrix of the volume.
    :type directions: tuple
    :param dicom_tags: Dictionary containing additional DICOM tags for the volume meta-data.
    :type dicom_tags: dict, optional
    :return: Dictionary containing the normalized meta-data for the volume.
    :rtype: dict

    :Usage example:

     .. code-block:: python

        import SimpleITK as sitk
        import supervisely as sly

        path = "/home/admin/work/volumes/vol_01.nrrd"

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(path)
        sitk_volume = reader.Execute()
        sitk_volume = _sitk_image_orient_ras(sitk_volume)
        dicom_tags = read_dicom_tags(paths[0], anonymize=anonymize)

        f_min_max = sitk.MinimumMaximumImageFilter()
        f_min_max.Execute(sitk_volume)
        meta = get_meta(
            sitk_volume.GetSize(),
            f_min_max.GetMinimum(),
            f_min_max.GetMaximum(),
            sitk_volume.GetSpacing(),
            sitk_volume.GetOrigin(),
            sitk_volume.GetDirection(),
            dicom_tags,
        )
    """

    # x = 1 - sagittal
    # y = 1 - coronal
    # z = 1 - axial
    volume_meta = normalize_volume_meta(
        {
            **dicom_tags,
            "channelsCount": 1,
            "rescaleSlope": 1,
            "rescaleIntercept": 0,
            "intensity": {
                "min": min_intensity,
                "max": max_intensity,
            },
            "dimensionsIJK": {
                "x": sitk_shape[0],
                "y": sitk_shape[1],
                "z": sitk_shape[2],
            },
            "ACS": "RAS",
            # instead of IJK2WorldMatrix field
            "spacing": spacing,
            "origin": origin,
            "directions": directions,
        }
    )
    return volume_meta


def inspect_nrrd_series(root_dir: str) -> List[str]:
    """
    Inspect a directory for NRRD series by recursively listing files with the ".nrrd" extension and returns a list of NRRD file paths found in the directory.

    :param root_dir: Directory to inspect for NRRD series.
    :type root_dir: str
    :return: List of NRRD file paths found in the given directory.
    :rtype: List[str]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "/home/admin/work/volumes/"
        nrrd_paths = sly.volume.inspect_nrrd_series(root_dir=path)
    """

    nrrd_paths = list_files_recursively(root_dir, [".nrrd"])
    logger.info(f"Total {len(nrrd_paths)} nnrd series in directory {root_dir}")
    return nrrd_paths


def read_nrrd_serie_volume(path: str) -> Tuple[sitk.Image, dict]:
    """
    Read NRRD volume with given path.

    :param path: Paths to DICOM volume files.
    :type path: List[str]
    :return: Volume data in SimpleITK.Image format and dictionary with metadata.
    :rtype: Tuple[SimpleITK.Image, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "/home/admin/work/volumes/vol_01.nrrd"
        sitk_volume, meta = sly.volume.read_nrrd_serie_volume(path)
    """

    import SimpleITK as sitk

    # find custom NRRD loader in gitlab supervisely_py/-/blob/feature/import-volumes/plugins/import/volumes/src/loaders/nrrd.py
    reader = sitk.ImageFileReader()
    # reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(path)
    sitk_volume = reader.Execute()

    sitk_volume = _sitk_image_orient_ras(sitk_volume)
    f_min_max = sitk.MinimumMaximumImageFilter()
    f_min_max.Execute(sitk_volume)
    meta = get_meta(
        sitk_volume.GetSize(),
        f_min_max.GetMinimum(),
        f_min_max.GetMaximum(),
        sitk_volume.GetSpacing(),
        sitk_volume.GetOrigin(),
        sitk_volume.GetDirection(),
        {},
    )
    return sitk_volume, meta


def read_nrrd_serie_volume_np(paths: List[str]) -> Tuple[np.ndarray, dict]:
    """
    Read NRRD volume with given path.

    :param path: Paths to NRRD volume file.
    :type path: List[str]
    :return: Volume data in NumPy array format and dictionary with metadata.
    :rtype: Tuple[np.ndarray, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "/home/admin/work/volumes/vol_01.nrrd"
        np_volume, meta = sly.volume.read_nrrd_serie_volume_np(path)
    """

    import SimpleITK as sitk

    sitk_volume, meta = read_nrrd_serie_volume(paths)
    volume_np = sitk.GetArrayFromImage(sitk_volume)
    volume_np = np.transpose(volume_np, (2, 1, 0))
    return volume_np, meta
