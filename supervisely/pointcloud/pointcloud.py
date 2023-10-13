# coding: utf-8
"""Functions for processing pointclouds"""

import os
import numpy as np
from typing import List, Optional
from supervisely._utils import is_development, abs_url
from supervisely.io.fs import ensure_base_path

# Do NOT use directly for extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_POINTCLOUD_EXTENSIONS = [".pcd"]


class PointcloudExtensionError(Exception):
    pass


class UnsupportedPointcloudFormat(Exception):
    pass


class PointcloudReadException(Exception):
    pass


def is_valid_ext(ext: str) -> bool:
    """
    Checks if given extention is supported.

    :param ext: Pointcloud file extension.
    :type ext: str
    :return: bool
    :rtype: :class:`bool`
    :Usage example:

    .. code-block:: python

        import supervisely as sly

        sly.pointcloud.is_valid_ext(".pcd")  # True
        sly.pointcloud.is_valid_ext(".mp4") # False
    """

    return ext.lower() in ALLOWED_POINTCLOUD_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    """
    Checks if file from given path with given extention is supported

    :param path: Pointcloud file path.
    :type path: str
    :return: bool
    :rtype: :class:`bool`
    :Usage example:

    .. code-block:: python

        import supervisely as sly

        path = "/Users/Downloads/demo_pointcloud-2/LYFT/1231201437602160096.pcd"
        sly.pointcloud.has_valid_ext(path)  # True
        sly.pointcloud.has_valid_ext(path) # False
    """

    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str) -> None:
    """
    Raise error if given extention is not supported

    :param ext: Pointcloud file extension.
    :type ext: str
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

    .. code-block:: python

        import supervisely as sly

        sly.pointcloud.validate_ext(".mp4")

        # UnsupportedPointcloudFormat: Unsupported pointcloud extension: .mp4.
        # Only the following extensions are supported: ['.pcd'].
    """

    if not is_valid_ext(ext):
        raise UnsupportedPointcloudFormat(
            "Unsupported pointcloud extension: {}. Only the following extensions are supported: {}.".format(
                ext, ALLOWED_POINTCLOUD_EXTENSIONS
            )
        )


def validate_format(path: str):
    """
    Raise error if file from given path with given extention is not supported

    :param path: Pointcloud file path.
    :type path: str
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "/Downloads/videos/111.mp4"
        sly.pointcloud.validate_format(path)

        # UnsupportedPointcloudFormat: Unsupported pointcloud extension: .mp4.
        # Only the following extensions are supported: ['.pcd'].
    """

    _, ext = os.path.splitext(path)
    validate_ext(ext)


def get_labeling_tool_url(dataset_id: int, pointcloud_id: int):
    """
    Get the URL for the labeling tool with the specified dataset ID and point cloud ID.

    :param dataset_id: Dataset ID in Supervisely.
    :type dataset_id: int
    :param pointcloud_id: Point cloud ID in Supervisely.
    :type pointcloud_id: int
    :return: URL for the labeling tool with the specified dataset ID and point cloud ID
    :rtype: str
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        pointcloud_id = 19373403
        pcd_info = api.pointcloud.get_info_by_id(pointcloud_id)
        url = sly.pointcloud.get_labeling_tool_url(pcd_info.dataset_id, pcd_info.id)

        print(url)
        # Output:
        # https://dev.supervise.ly/app/point-clouds/?datasetId=55875&pointCloudId=19373403
    """

    res = f"/app/point-clouds/?datasetId={dataset_id}&pointCloudId={pointcloud_id}"
    if is_development():
        res = abs_url(res)
    return res


def get_labeling_tool_link(url, name="open in labeling tool"):
    """
    Get HTML link to the labeling tool with the specified URL and name.

    :param url: URL of the labeling tool.
    :type url: str
    :param name: Name of the link, default is "open in labeling tool".
    :type name: str
    :return: HTML link to the labeling tool with the specified URL and name.
    :rtype: str
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        pointcloud_id = 19373403
        pcd_info = api.pointcloud.get_info_by_id(pointcloud_id)
        url = sly.pointcloud.get_labeling_tool_url(pcd_info.dataset_id, pcd_info.id)
        name = "my link"

        link = sly.pointcloud.get_labeling_tool_link(url, name)

        print(link)
        # Output:
        # <a
        #     href="https://dev.supervise.ly/app/point-clouds/?datasetId=55875&pointCloudId=19373403"
        #     rel="noopener noreferrer"
        #     target="_blank"
        # >
        #     my link<i class="zmdi zmdi-open-in-new" style="margin-left: 5px"></i>
        # </a>
    """

    return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{name}<i class="zmdi zmdi-open-in-new" style="margin-left: 5px"></i></a>'


def read(path: str, coords_dims: Optional[List[int]] = None) -> np.ndarray:
    """
    Loads a pointcloud from the specified file and returns it in XYZ format.

    :param path: Path to file.
    :type path: str
    :return: Numpy array
    :rtype: :class:`np.ndarray`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        pcd_np = sly.pointcloud.read('/home/admin/work/pointclouds/ptc0.pcd')
    """

    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "No module named open3d. Please make sure that module is installed from pip and try again."
        )
    validate_format(path)
    if coords_dims is None:
        coords_dims = [0, 1, 2]
    pcd_data = o3d.io.read_point_cloud(path)
    if pcd_data is None:
        raise IOError(f"open3d can not open the file {path}")
    pointcloud_np = np.asarray(pcd_data.points)
    pointcloud_np = pointcloud_np[:, coords_dims]
    return pointcloud_np


def write(path: str, pointcloud_np: np.ndarray, coords_dims: Optional[List[int]] = None) -> bool:
    """
    Saves a pointcloud to the specified file. It creates directory from path if the directory for this path does not exist.

    :param path: Path to file.
    :type path: str
    :param pointcloud_np: Pointcloud [N, 3] in XYZ format.
    :type pointcloud_np: :class:`np.ndarray`
    :param coords_dims: List of indexes for (X, Y, Z) coords. Default (if None): [0, 1, 2].
    :type coords_dims: Optional[List[int]]
    :return: Success or not.
    :rtype: bool
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        import numpy as np

        pointcloud = np.random.randn(100, 3)

        ptc = sly.pointcloud.write('/home/admin/work/pointclouds/ptc0.pcd', pointcloud)
    """

    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "No module named open3d. Please make sure that module is installed from pip and try again."
        )
    ensure_base_path(path)
    validate_format(path)
    if coords_dims is None:
        coords_dims = [0, 1, 2]
    pointcloud_np = pointcloud_np[:, coords_dims]
    pcd_data = o3d.geometry.PointCloud()
    pcd_data.points = o3d.utility.Vector3dVector(pointcloud_np)
    return o3d.io.write_point_cloud(path, pcd_data)
