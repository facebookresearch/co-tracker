from typing import Callable, List, Optional, Union

from tqdm import tqdm

from supervisely import get_project_class
from supervisely.api.api import Api
from supervisely.project.pointcloud_episode_project import (
    download_pointcloud_episode_project,
)
from supervisely.project.pointcloud_project import download_pointcloud_project
from supervisely.project.project import download_project
from supervisely.project.project_type import ProjectType
from supervisely.project.video_project import download_video_project
from supervisely.project.volume_project import download_volume_project


def download(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    **kwargs,
) -> None:
    """
    Downloads project of any type to the local directory. See methods `sly.download_project`,
    `sly.download_video_project`, `sly.download_volume_project`, `sly.download_pointcloud_project`,
    `sly.download_pointcloud_episode_project` to examine full list of possible arguments.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID, which will be downloaded.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param log_progress: Show downloading logs in the output.
    :type log_progress: bool, optional
    :param progress_cb: Function for tracking download progress.
    :type progress_cb: tqdm or callable, optional

    :return: None.
    :rtype: NoneType
    :Usage example:

    .. code-block:: python

        import os
        from dotenv import load_dotenv

        from tqdm import tqdm
        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        dest_dir = 'your/local/dest/dir'

        # Download image project
        project_id_image = 17732
        project_info = api.project.get_info_by_id(project_id_image)
        num_images = project_info.items_count

        p = tqdm(desc="Downloading image project", total=num_images)
        sly.download(
            api,
            project_id_image,
            dest_dir,
            progress_cb=p,
            save_image_info=True,
            save_images=True,
        )

        # Download video project
        project_id_video = 60498
        project_info = api.project.get_info_by_id(project_id_video)
        num_videos = project_info.items_count

        p = tqdm(desc="Downloading video project", total=num_videos)
        sly.download(
            api,
            project_id_video,
            dest_dir,
            progress_cb=p,
            save_video_info=True,
        )

        # Download volume project
        project_id_volume = 18594
        project_info = api.project.get_info_by_id(project_id_volume)
        num_volumes = project_info.items_count

        p = tqdm(desc="Downloading volume project",total=num_volumes)
        sly.download(
            api,
            project_id_volume,
            dest_dir,
            progress_cb=p,
            download_volumes=True,
        )

        # Download pointcloud project
        project_id_ptcl = 18592
        project_info = api.project.get_info_by_id(project_id_ptcl)
        num_ptcl = project_info.items_count

        p = tqdm(desc="Downloading pointcloud project", total=num_ptcl)
        sly.download(
            api,
            project_id_ptcl,
            dest_dir,
            progress_cb=p,
            download_pointclouds_info=True,
        )

        # Download some datasets from pointcloud episodes project
        project_id_ptcl_ep = 18593
        dataset_ids = [43546, 45765, 45656]

        p = tqdm(
            desc="Download some datasets from pointcloud episodes project",
            total=len(dataset_ids),
        )
        sly.download(
            api,
            project_id_ptcl_ep,
            dest_dir,
            dataset_ids,
            progress_cb=p,
            download_pcd=True,
            download_related_images=True,
            download_annotations=True,
            download_pointclouds_info=True,
        )
    """

    project_info = api.project.get_info_by_id(project_id)

    project_class = get_project_class(project_info.type)

    project_class.download(
        api=api,
        project_id=project_id,
        dest_dir=dest_dir,
        dataset_ids=dataset_ids,
        log_progress=log_progress,
        progress_cb=progress_cb,
        **kwargs,
    )
