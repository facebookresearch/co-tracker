import json
import os
import mimetypes
from os.path import basename, join

import supervisely.io.env as env
from supervisely._utils import is_production
from supervisely.api.api import Api
from supervisely.app.fastapi import get_name_from_env
from supervisely.io.fs import get_file_name_with_ext, silent_remove, archive_directory, remove_dir
import supervisely.io.env as sly_env
from supervisely import rand_str
from supervisely.task.progress import Progress
from supervisely.sly_logger import logger
from supervisely.team_files import RECOMMENDED_EXPORT_PATH


def set_project(id: int):
    if is_production() is True:
        api = Api()
        task_id = sly_env.task_id()
        api.task.set_output_project(task_id, project_id=id)
    else:
        print(f"Output project: id={id}")


def set_directory(teamfiles_dir: str):
    """
    Sets a link to a teamfiles directory in workspace tasks interface
    """
    if is_production():
        api = Api()
        task_id = sly_env.task_id()

        if api.task.get_info_by_id(task_id) is None:
            raise KeyError(
                f"Task with ID={task_id} is either not exist or not found in your account"
            )

        team_id = api.task.get_info_by_id(task_id)["teamId"]

        if api.team.get_info_by_id(team_id) is None:
            raise KeyError(
                f"Team with ID={team_id} is either not exist or not found in your account"
            )

        files = api.file.list2(team_id, teamfiles_dir, recursive=True)

        # if directory is empty or not exists
        if len(files) == 0:
            # some data to create dummy .json file to get file id
            data = {"team_id": team_id, "task_id": task_id, "directory": teamfiles_dir}
            filename = f"{rand_str(10)}.json"

            src_path = os.path.join("/tmp/", filename)
            with open(src_path, "w") as f:
                json.dump(data, f)

            dst_path = os.path.join(teamfiles_dir, filename)
            file_id = api.file.upload(team_id, src_path, dst_path).id

            silent_remove(src_path)

        else:
            file_id = files[0].id

        api.task.set_output_directory(task_id, file_id, teamfiles_dir)

    else:
        print(f"Output directory: '{teamfiles_dir}'")


def set_download(local_path: str):
    """
    Receives a path to the local file or directory. If the path is a directory, it will be archived before uploading.
    After sets a link to a uploaded file in workspace tasks interface according to the file type.
    If the file is an archive, the set_output_archive method is called and "Download archive" text is displayed.
    If the file is not an archive, the set_output_file_download method is called and "Download file" text is displayed.

    :param local_path: path to the local file or directory, which will be uploaded to the teamfiles
    :type local_path: str
    :return: None
    :rtype: None
    """
    if os.path.isdir(local_path):
        archive_path = f"{local_path}.tar"
        archive_directory(local_path, archive_path)
        remove_dir(local_path)
        local_path = archive_path

    if is_production():
        api = Api()
        task_id = sly_env.task_id()
        upload_progress = []

        team_id = env.team_id()

        def _print_progress(monitor, upload_progress):
            if len(upload_progress) == 0:
                upload_progress.append(
                    Progress(
                        message=f"Uploading '{basename(local_path)}'",
                        total_cnt=monitor.len,
                        ext_logger=logger,
                        is_size=True,
                    )
                )
            upload_progress[0].set_current_value(monitor.bytes_read)

        def _is_archive(local_path: str) -> bool:
            """
            Checks if the file is an archive by its mimetype using list of the most common archive mimetypes.

            :param local_path: path to the local file
            :type local_path: str
            :return: True if the file is an archive, False otherwise
            :rtype: bool
            """
            archive_mimetypes = [
                "application/zip",
                "application/x-tar",
                "application/x-gzip",
                "application/x-bzip2",
                "application/x-7z-compressed",
                "application/x-rar-compressed",
                "application/x-xz",
                "application/x-lzip",
                "application/x-lzma",
                "application/x-lzop",
                "application/x-bzip",
                "application/x-bzip2",
                "application/x-compress",
                "application/x-compressed",
            ]

            return mimetypes.guess_type(local_path)[0] in archive_mimetypes

        remote_path = join(
            RECOMMENDED_EXPORT_PATH,
            get_name_from_env(),
            str(task_id),
            f"{get_file_name_with_ext(local_path)}",
        )
        file_info = api.file.upload(
            team_id=team_id,
            src=local_path,
            dst=remote_path,
            progress_cb=lambda m: _print_progress(m, upload_progress),
        )

        if _is_archive(local_path):
            api.task.set_output_archive(task_id, file_info.id, file_info.name)
        else:
            api.task.set_output_file_download(task_id, file_info.id, file_info.name)

        logger.info(f"Remote file: id={file_info.id}, name={file_info.name}")
        silent_remove(local_path)

    else:
        print(f"Output file: '{local_path}'")
