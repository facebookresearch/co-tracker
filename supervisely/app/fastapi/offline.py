import functools
import os
import pathlib
import shutil
import tempfile
import threading
import traceback
from collections import namedtuple
from distutils.dir_util import copy_tree

from fastapi import FastAPI
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

import supervisely as sly


def get_static_paths_by_mounted_object(mount) -> list:
    StaticPath = namedtuple("StaticPath", ["local_path", "url_path"])
    static_paths = []

    if hasattr(mount, "routes"):
        for current_route in mount.routes:
            if type(current_route) == Mount and type(current_route.app) == FastAPI:
                all_children_paths = get_static_paths_by_mounted_object(current_route)
                for index, current_path in enumerate(all_children_paths):
                    current_url_path = pathlib.Path(
                        str(current_route.path).lstrip("/"),
                        str(current_path.url_path).lstrip("/"),
                    )
                    all_children_paths[index] = StaticPath(
                        local_path=current_path.local_path, url_path=current_url_path
                    )
                static_paths.extend(all_children_paths)
            elif (
                type(current_route) == Mount and type(current_route.app) == StaticFiles
            ):
                static_paths.append(
                    StaticPath(
                        local_path=pathlib.Path(current_route.app.directory),
                        url_path=pathlib.Path(str(current_route.path).lstrip("/")),
                    )
                )

    return static_paths


def dump_statics_to_dir(static_dir_path: pathlib.Path, static_paths: list):
    for current_path in static_paths:
        current_local_path: pathlib.Path = current_path.local_path
        current_url_path: pathlib.Path = static_dir_path / current_path.url_path

        def _filter_static_files(path: pathlib.Path):
            extensions_to_delete = ['.py', '.pyc', '.md', '.sh']
            for dirpath, _, filenames in os.walk(path.as_posix()):

                if filenames:
                    for file in filenames:
                        if os.path.splitext(os.path.basename(file))[1] in extensions_to_delete:
                            filepath = pathlib.Path(dirpath, file)
                            if pathlib.Path.exists(filepath):
                                pathlib.Path.unlink(filepath)

        if current_local_path.is_dir():
            current_url_path.mkdir(parents=True, exist_ok=True)
            copy_tree(
                current_local_path.as_posix(),
                current_url_path.as_posix(),
                preserve_symlinks=True,
            )

            _filter_static_files(current_url_path)


def dump_html_to_dir(static_dir_path, template):
    pathlib.Path(static_dir_path / template.template.name).write_bytes(template.body)


def get_offline_session_files_path(task_id) -> pathlib.Path:
    return pathlib.Path("/", "offline-sessions", str(task_id), "app-template")


def upload_to_supervisely(static_dir_path):
    api: sly.Api = sly.Api.from_env()

    team_id = sly.env.team_id()
    task_id = sly.env.task_id(raise_not_found=False)
    task_id = 0000 if task_id is None else task_id
    remote_dir = get_offline_session_files_path(task_id)

    res_remote_dir: str = api.file.upload_directory(
        team_id=team_id,
        local_dir=static_dir_path.as_posix(),
        remote_dir=remote_dir.as_posix(),
        change_name_if_conflict=False,
        replace_if_conflict=True
    )

    if os.getenv("TASK_ID") is not None:
        api.task.update_meta(id=task_id, data={"templateRootDirectory": res_remote_dir})

    sly.logger.info(f"App files stored in {res_remote_dir} for offline usage")


def dump_files_to_supervisely(app: FastAPI, template_response):
    try:
        if os.getenv("TASK_ID") is None:
            sly.logger.debug(
                f"Debug mode: saving app files for offline usage is skipped"
            )
            return

        if (
            os.getenv("_SUPERVISELY_OFFLINE_FILES_UPLOADED", "False") == "True"
            and template_response.context.get("request") is not None
        ):
            return
        os.environ["_SUPERVISELY_OFFLINE_FILES_UPLOADED"] = "True"
        sly.logger.info(f"Saving app files for offline usage")

        app_template_path = pathlib.Path(tempfile.mkdtemp())
        app_static_paths = get_static_paths_by_mounted_object(mount=app)
        dump_statics_to_dir(
            static_dir_path=app_template_path, static_paths=app_static_paths
        )
        dump_html_to_dir(static_dir_path=app_template_path, template=template_response)

        upload_to_supervisely(static_dir_path=app_template_path)

        shutil.rmtree(app_template_path.as_posix())

    except Exception as ex:
        traceback.print_exc()
        sly.logger.warning(f"Cannot dump files for offline usage, reason: {ex}")
        os.environ["_SUPERVISELY_OFFLINE_FILES_UPLOADED"] = "False"


def available_after_shutdown(app: FastAPI):
    def func_layer_wrapper(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            template_response = f(*args, **kwargs)
            try:
                if sly.utils.is_production():
                    sly.logger.info(f"Start dumping app UI for offline mode")
                    threading.Thread(
                        target=functools.partial(
                            dump_files_to_supervisely, app, template_response
                        ),
                        daemon=False,
                    ).start()

            except Exception as ex:
                traceback.print_exc()
                sly.logger.warning(f"Cannot dump files for offline usage, reason: {ex}")

            finally:
                return template_response

        return wrapper

    return func_layer_wrapper
