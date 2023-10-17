# coding: utf-8
import os
from re import L
from typing import Callable, List, Optional


RAISE_IF_NOT_FOUND = True


def flag_from_env(s):
    return s.upper() in ["TRUE", "YES", "1"]


def remap_gpu_devices(in_device_ids):
    """
    Working limitation for CUDA
    :param in_device_ids: real GPU devices indexes. e.g.: [3, 4, 7]
    :return: CUDA ordered GPU indexes, e.g.: [0, 1, 2]
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, in_device_ids))
    return list(range(len(in_device_ids)))


def _int_from_env(value):
    if value is None:
        return value
    return int(value)


def _parse_from_env(
    name: str,
    keys: List[str],
    postprocess_fn: Callable,
    default=None,
    raise_not_found=False,
):
    for k in keys:
        if k in os.environ:
            return postprocess_fn(os.environ[k])

    # env not found
    if raise_not_found is True:
        raise KeyError(
            f"{name} is not defined as environment variable. One of the envs has to be defined: {keys}. Learn more in developer portal: https://developer.supervise.ly/getting-started/environment-variables"
        )

    return default


def agent_id(raise_not_found=True):
    return _parse_from_env(
        name="agent_id",
        keys=["AGENT_ID"],
        postprocess_fn=_int_from_env,
        default=None,
        raise_not_found=raise_not_found,
    )


def agent_storage(raise_not_found=True):
    return _parse_from_env(
        name="agent_storage",
        keys=["AGENT_STORAGE"],
        postprocess_fn=lambda x: x,
        default=None,
        raise_not_found=raise_not_found,
    )


def team_id(raise_not_found=True):
    return _parse_from_env(
        name="team_id",
        keys=["CONTEXT_TEAMID", "context.teamId", "TEAM_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def workspace_id(raise_not_found=True):
    return _parse_from_env(
        name="workspace_id",
        keys=["CONTEXT_WORKSPACEID", "context.workspaceId", "WORKSPACE_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def project_id(raise_not_found=True):
    return _parse_from_env(
        name="project_id",
        keys=[
            "CONTEXT_PROJECTID",
            "context.projectId",
            "modal.state.slyProjectId",
            "PROJECT_ID",
            "modal.state.inputProjectId",
        ],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def dataset_id(raise_not_found=True):
    return _parse_from_env(
        name="dataset_id",
        keys=[
            "CONTEXT_DATASETID",
            "context.datasetId",
            "modal.state.slyDatasetId",
            "DATASET_ID",
            "modal.state.inputDatasetId",
        ],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def team_files_folder(raise_not_found=True):
    return _parse_from_env(
        name="team_files_folder",
        keys=[
            "CONTEXT_SLYFOLDER",
            "context.slyFolder",
            "modal.state.slyFolder",
            "FOLDER",
        ],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def folder(raise_not_found=True):
    return team_files_folder(raise_not_found)


def team_files_file(raise_not_found=True):
    return _parse_from_env(
        name="team_files_file",
        keys=["CONTEXT_SLYFILE", "context.slyFile", "modal.state.slyFile", "FILE"],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def server_address(raise_not_found=True):
    return _parse_from_env(
        name="server_address",
        keys=["SERVER_ADDRESS"],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def api_token(raise_not_found=True):
    return _parse_from_env(
        name="api_token",
        keys=["API_TOKEN"],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def file(raise_not_found=True):
    return team_files_file(raise_not_found)


def task_id(raise_not_found=True):
    return _parse_from_env(
        name="task_id",
        keys=["TASK_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def user_login(raise_not_found=True):
    return _parse_from_env(
        name="user_login",
        keys=["USER_LOGIN", "context.userLogin", "CONTEXT_USERLOGIN"],
        postprocess_fn=lambda x: int(x),
        default="user (debug)",
        raise_not_found=raise_not_found,
    )


def app_name(raise_not_found=True):
    return _parse_from_env(
        name="app_name",
        keys=["APP_NAME"],
        postprocess_fn=lambda x: int(x),
        default="Supervisely App (debug)",
        raise_not_found=raise_not_found,
    )


def user_id(raise_not_found=True):
    return _parse_from_env(
        name="user_id",
        keys=["USER_ID", "context.userId", "CONTEXT_USERID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def content_origin_update_interval():
    return _parse_from_env(
        name="content_origin_update_interval",
        keys=["CONTENT_ORIGIN_UPDATE_INTERVAL"],
        postprocess_fn=lambda x: float(x),
        default=0.5,
        raise_not_found=False,
    )


def smart_cache_ttl(raise_not_found=False, default=120):
    return _parse_from_env(
        name="smart_cache_ttl",
        keys=["SMART_CACHE_TTL"],
        postprocess_fn=lambda x: max(int(x), 1),
        default=default,
        raise_not_found=raise_not_found,
    )


def smart_cache_size(raise_not_found=False, default=256):
    return _parse_from_env(
        name="smart_cache_size",
        keys=["SMART_CACHE_SIZE"],
        postprocess_fn=lambda x: max(int(x), 1),
        default=default,
        raise_not_found=raise_not_found,
    )


def smart_cache_container_dir(default="/tmp/smart_cache"):
    return _parse_from_env(
        name="smart_cache_container_dir",
        keys=["SMART_CACHE_CONTAINER_DIR"],
        default=default,
        raise_not_found=False,
        postprocess_fn=lambda x: x.strip(),
    )


def autostart():
    return _parse_from_env(
        name="autostart",
        keys=["modal.state.autostart"],
        default=False,
        raise_not_found=False,
        postprocess_fn=flag_from_env,
    )


def set_autostart(value: Optional[str]):
    """
    Set modal.state.autostart env.
    Possible values (case insensetive): "1", "true", "yes".
    Use `value=None`, to remove variable.
    """
    if value is None:
        os.environ.pop("modal.state.autostart", None)
        return

    if not flag_from_env(value):
        raise ValueError("Unknown value for `autostart` env. Use `1`, `true`, `yes` or None.")
    os.environ["modal.state.autostart"] = value
