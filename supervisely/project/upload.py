from typing import Callable, List, Optional, Union

from tqdm import tqdm

from supervisely.api.api import Api
from supervisely.project import read_project
from supervisely.project.pointcloud_episode_project import (
    upload_pointcloud_episode_project,
)
from supervisely.project.pointcloud_project import upload_pointcloud_project
from supervisely.project.project import upload_project
from supervisely.project.project_type import ProjectType
from supervisely.project.video_project import upload_video_project
from supervisely.project.volume_project import upload_volume_project
from supervisely import get_project_class


def upload(
    src_dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[None] = None,
    log_progress: Optional[bool] = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    **kwargs,
) -> None:
    """
    Uploads project of any type from the local directory. See methods `sly.upload_project`,
    `sly.upload_video_project`, `sly.upload_volume_project`, `sly.upload_pointcloud_project`,
    `sly.upload_pointcloud_episode_project` to examine full list of possible arguments.

    :param src_dir: Source path to local directory.
    :type src_dir: str
    :param api: Supervisely API address and token.
    :type api: Api
    :param workspace_id: Destination workspace ID.
    :type workspace_id: int
    :param project_name: Custom project name. By default, it's a directory name.
    :type project_name: str, optional
    :param log_progress: Show uploading logs in the output.
    :type log_progress: bool, optional
    :param progress_cb: Function for tracking upload progress.
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

        src_dir = '/your/local/source/dir'

        # Upload image project
        project_fs = sly.read_project(src_dir)

        pbar = tqdm(desc="Uploading image project", total=project_fs.total_items)
        sly.upload(src_dir, api, workspace_id, project_name, progress_cb=pbar)

        # Upload video project
        sly.upload(
            src_dir,
            api,
            workspace_id,
            project_name="Some Video Project",
            log_progress=True,
            include_custom_data=True
        )

        # Upload volume project
        sly.upload(src_dir, api, workspace_id, project_name="Some Volume Project", log_progress=True)

        # Upload pointcloud project
        project_fs = read_project(directory)

        pbar = tqdm(desc="Uploading pointcloud project", total=project_fs.total_items)
        sly.upload(
            src_dir,
            api,
            workspace_id,
            project_name="Some Pointcloud Project",
            progress_cb=pbar,
        )

        # Upload pointcloud episodes project
        project_fs = read_project(src_dir)

        with tqdm(desc="Upload pointcloud episodes project", total=project_fs.total_items) as pbar:
            sly.upload(
                src_dir,
                api,
                workspace_id,
                project_name="Some Pointcloud Episodes Project",
                progress_cb=pbar,
            )
    """

    project_fs = read_project(src_dir)

    if progress_cb is not None:
        log_progress = False
        kwargs["progress_cb"] = progress_cb

    if progress_cb is not None and project_fs.meta.project_type in (
        ProjectType.VIDEOS.value,
        ProjectType.VOLUMES.value,
    ):
        log_progress = True

    project_class = get_project_class(project_fs.meta.project_type)

    project_class.upload(
        src_dir,
        api=api,
        workspace_id=workspace_id,
        project_name=project_name,
        log_progress=log_progress,
        **kwargs,
    )
