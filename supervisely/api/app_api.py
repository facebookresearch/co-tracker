# coding: utf-8
from __future__ import annotations

import os
import json
from typing import NamedTuple, List, Dict, Optional
from supervisely.api.module_api import ApiField
from supervisely.api.task_api import TaskApi
from supervisely._utils import take_with_default

# from supervisely.app.constants import DATA, STATE, CONTEXT, TEMPLATE
STATE = "state"
DATA = "data"
TEMPLATE = "template"

from supervisely.io.fs import ensure_base_path
from supervisely.task.progress import Progress
from supervisely._utils import sizeof_fmt
from supervisely import logger

_context_menu_targets = {
    "files_folder": {
        "help": "Context menu of folder in Team Files. Target value is directory path.",
        "type": str,
        "key": "slyFolder",
    },
    "files_file": {
        "help": "Context menu of file in Team Files. Target value is file path.",
        "type": str,
        "key": "slyFile",
    },
    "images_project": {
        "help": "Context menu of images project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "images_dataset": {
        "help": "Context menu of images dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "videos_project": {
        "help": "Context menu of videos project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "videos_dataset": {
        "help": "Context menu of videos dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "point_cloud_episodes_project": {
        "help": "Context menu of pointcloud episodes project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "point_cloud_episodes_dataset": {
        "help": "Context menu of pointcloud episodes dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "point_cloud_project": {
        "help": "Context menu of pointclouds project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "point_cloud_dataset": {
        "help": "Context menu of pointclouds dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "volumes_project": {
        "help": "Context menu of volumes project (DICOMs). Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "volumes_dataset": {
        "help": "Context menu of volumes dataset (DICOMs). Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "team": {
        "help": "Context menu of team. Target value is team id.",
        "type": int,
        "key": "slyTeamId",
    },
    "team_member": {
        "help": "Context menu of team member. Target value is user id.",
        "type": int,
        "key": "slyMemberId",
    },
    "labeling_job": {
        "help": "Context menu of labeling job. Target value is labeling job id.",
        "type": int,
        "key": "slyJobId",
    },
    "ecosystem": {
        "help": "Run button in ecosystem. It is not needed to define any target",
        "key": "nothing",
    },
}


class AppInfo(NamedTuple):
    """AppInfo"""

    id: int
    created_by_id: int
    module_id: int
    disabled: bool
    user_login: str
    config: dict
    name: str
    slug: str
    is_shared: bool
    tasks: List[Dict]
    repo: str
    team_id: int


class ModuleInfo(NamedTuple):
    """ModuleInfo in Ecosystem"""

    id: int
    slug: str
    name: str
    type: str
    config: dict
    readme: str
    repo: str
    github_repo: str
    meta: dict
    created_at: str
    updated_at: str

    @staticmethod
    def from_json(data: dict) -> ModuleInfo:
        info = ModuleInfo(
            id=data["id"],
            slug=data["slug"],
            name=data["name"],
            type=data["type"],
            config=data["config"],
            readme=data.get("readme"),
            repo=data.get("repositoryModuleUrl"),
            github_repo=data.get("repo"),
            meta=data.get("meta"),
            created_at=data["createdAt"],
            updated_at=data["updatedAt"],
        )
        if "contextMenu" in info.config:
            info.config["context_menu"] = info.config["contextMenu"]
        return info

    def get_latest_release(self, default=""):
        release = self.meta.get("releases", [default])[0]
        return release

    def arguments_help(self):
        modal_args = self.get_modal_window_arguments()
        if len(modal_args) == 0:
            print(
                f"App '{self.name}' has no additional options \n"
                "that can be configured manually in modal dialog window \n"
                "before running app."
            )
        else:
            print(
                f"App '{self.name}' has additional options "
                "that can be configured manually in modal dialog window before running app. "
                "You can change them or keep defaults: "
            )
            print(json.dumps(modal_args, sort_keys=True, indent=4))

        targets = self.get_context_menu_targets()

        if len(targets) > 0:
            print("App has to be started from the context menus:")
            for target in targets:
                print(
                    f'{target} : {_context_menu_targets.get(target, {"help": "empty description"})["help"]}'
                )
            print(
                "It is needed to call get_arguments method with defined target argument (pass one of the values above)."
            )

        if "ecosystem" in targets:
            pass

    def get_modal_window_arguments(self):
        params = self.config.get("modalTemplateState", {})
        return params

    def get_arguments(self, **kwargs) -> dict:
        params = self.config.get("modalTemplateState", {})
        targets = self.get_context_menu_targets()
        if len(targets) > 0 and len(kwargs) == 0 and "ecosystem" not in targets:
            raise ValueError(
                "target argument has to be defined. Call method 'arguments_help' to print help info for developer"
            )
        if len(kwargs) > 1:
            raise KeyError("Only one target is allowed")
        if len(kwargs) == 1:
            # params["state"] = {}
            for target_key, target_value in kwargs.items():
                if target_key not in targets:
                    raise KeyError(
                        f"You passed {target_key}, but allowed only one of the targets: {targets}"
                    )
                key = _context_menu_targets[target_key]["key"]
                valid_type = _context_menu_targets[target_key]["type"]
                if type(target_value) is not valid_type:
                    raise ValueError(
                        f"Target {target_key} has value {target_value} of type {type(target_value)}. Allowed type is {valid_type}"
                    )
                params[key] = target_value
        return params

    def get_context_menu_targets(self):
        if "context_menu" in self.config:
            if "target" in self.config["context_menu"]:
                return self.config["context_menu"]["target"]
        return []


class SessionInfo(NamedTuple):
    """SessionInfo"""

    task_id: int
    user_id: int
    module_id: int  # in ecosystem
    app_id: int  # in team (recent apps)

    details: dict

    @staticmethod
    def from_json(data: dict) -> SessionInfo:
        # {'taskId': 21012, 'userId': 6, 'moduleId': 83, 'appId': 578}

        if "meta" in data:
            info = SessionInfo(
                task_id=data["id"],
                user_id=data["createdBy"],
                module_id=data["moduleId"],
                app_id=data["meta"]["app"]["id"],
                details=data,
            )
        else:
            info = SessionInfo(
                task_id=data["taskId"],
                user_id=data["userId"],
                module_id=data["moduleId"],
                app_id=data["appId"],
                details={},
            )
        return info


class AppApi(TaskApi):
    """AppApi"""

    @staticmethod
    def info_sequence():
        """info_sequence"""
        return [
            ApiField.ID,
            ApiField.CREATED_BY_ID,
            ApiField.MODULE_ID,
            ApiField.DISABLED,
            ApiField.USER_LOGIN,
            ApiField.CONFIG,
            ApiField.NAME,
            ApiField.SLUG,
            ApiField.IS_SHARED,
            ApiField.TASKS,
            ApiField.REPO,
            ApiField.TEAM_ID,
        ]

    @staticmethod
    def info_tuple_name():
        """info_tuple_name"""
        return "AppInfo"

    def _convert_json_info(self, info: dict, skip_missing=True) -> AppInfo:
        """_convert_json_info"""
        res = super(TaskApi, self)._convert_json_info(info, skip_missing=skip_missing)
        return AppInfo(**res._asdict())

    def get_info_by_id(self, id: int) -> AppInfo:
        """
        :param id: int
        :return: application info by numeric id
        """
        return self._get_info_by_id(id, "apps.info")

    def get_list(
        self,
        team_id,
        filter=None,
        context=None,
        repository_key=None,
        show_disabled=False,
        integrated_into=None,
        session_tags=None,
        only_running=False,
        with_shared=True,
    ) -> List[AppInfo]:
        """get_list"""

        return self.get_list_all_pages(
            method="apps.list",
            data={
                "teamId": team_id,
                "filter": take_with_default(
                    filter, []
                ),  # for example [{"field": "id", "operator": "=", "value": None}]
                "context": take_with_default(context, []),  # for example ["images_project"]
                "repositoryKey": repository_key,
                "integratedInto": take_with_default(
                    integrated_into, []
                ),  # for example ["image_annotation_tool"]
                "sessionTags": take_with_default(session_tags, []),  # for example ["string"]
                "onlyRunning": only_running,
                "showDisabled": show_disabled,
                "withShared": with_shared,
            },
        )

    def run_dtl(self, workspace_id, dtl_graph, agent_id=None):
        """run_dtl"""
        raise RuntimeError("Method is unavailable")

    def _run_plugin_task(
        self,
        task_type,
        agent_id,
        plugin_id,
        version,
        config,
        input_projects,
        input_models,
        result_name,
    ):
        """_run_plugin_task"""
        raise RuntimeError("Method is unavailable")

    def run_train(
        self,
        agent_id,
        input_project_id,
        input_model_id,
        result_nn_name,
        train_config=None,
    ):
        """run_train"""
        raise RuntimeError("Method is unavailable")

    def run_inference(
        self,
        agent_id,
        input_project_id,
        input_model_id,
        result_project_name,
        inference_config=None,
    ):
        """run_inference"""
        raise RuntimeError("Method is unavailable")

    def get_training_metrics(self, task_id):
        """get_training_metrics"""
        raise RuntimeError("Method is unavailable")

    def deploy_model(self, agent_id, model_id):
        """deploy_model"""
        raise RuntimeError("Method is unavailable")

    def get_import_files_list(self, id):
        """get_import_files_list"""
        raise RuntimeError("Method is unavailable")

    def download_import_file(self, id, file_path, save_path):
        """download_import_file"""
        raise RuntimeError("Method is unavailable")

    def create_task_detached(self, workspace_id, task_type: str = None):
        """create_task_detached"""
        raise RuntimeError("Method is unavailable")

    def upload_files(self, task_id, abs_paths, names, progress_cb=None):
        """upload_files"""
        raise RuntimeError("Method is unavailable")

    def initialize(self, task_id, template, data=None, state=None):
        """initialize"""
        d = take_with_default(data, {})
        if "notifyDialog" not in d:
            d["notifyDialog"] = None
        if "scrollIntoView" not in d:
            d["scrollIntoView"] = None

        s = take_with_default(state, {})
        fields = [
            {"field": TEMPLATE, "payload": template},
            {"field": DATA, "payload": d},
            {"field": STATE, "payload": s},
        ]
        resp = self._api.task.set_fields(task_id, fields)
        return resp

    def get_url(self, task_id):
        """get_url"""
        return f"/apps/sessions/{task_id}"

    def download_git_file(self, app_id, version, file_path, save_path):
        """download_git_file"""
        raise NotImplementedError()

    def download_git_archive(
        self,
        ecosystem_item_id,
        app_id,
        version,
        save_path,
        log_progress=True,
        ext_logger=None,
    ):
        """download_git_archive"""
        payload = {
            ApiField.ECOSYSTEM_ITEM_ID: ecosystem_item_id,
            ApiField.VERSION: version,
            "isArchive": True,
        }
        if app_id is not None:
            payload[ApiField.APP_ID] = app_id

        response = self._api.post("ecosystem.file.download", payload, stream=True)
        if log_progress:
            if ext_logger is None:
                ext_logger = logger

            length = -1
            # Content-Length
            if "Content-Length" in response.headers:
                length = int(response.headers["Content-Length"])
            progress = Progress("Downloading: ", length, ext_logger=ext_logger, is_size=True)

        mb1 = 1024 * 1024
        ensure_base_path(save_path)
        with open(save_path, "wb") as fd:
            log_size = 0
            for chunk in response.iter_content(chunk_size=mb1):
                fd.write(chunk)
                log_size += len(chunk)
                if log_progress and log_size > mb1:
                    progress.iters_done_report(log_size)
                    log_size = 0

    def get_info(self, module_id, version=None):
        """get_info"""
        data = {ApiField.ID: module_id}
        if version is not None:
            data[ApiField.VERSION] = version
        response = self._api.post("ecosystem.info", data)
        return response.json()

    def get_ecosystem_module_info(self, module_id, version=None) -> ModuleInfo:
        """get_module_info"""
        data = {ApiField.ID: module_id}
        if version is not None:
            data[ApiField.VERSION] = version
        response = self._api.post("ecosystem.info", data)
        return ModuleInfo.from_json(response.json())

    def get_ecosystem_module_id(self, slug: str):
        modules = self.get_list_all_pages(
            method="ecosystem.list",
            data={"filter": [{"field": "slug", "operator": "=", "value": slug}]},
            convert_json_info_cb=lambda x: x,
        )
        if len(modules) == 0:
            raise KeyError(f"Module {slug} not found in ecosystem")
        if len(modules) > 1:
            raise KeyError(
                f"Ecosystem is broken: there are {len(modules)} modules with the same slug: {slug}. Please, contact tech support"
            )
        return modules[0]["id"]

    def get_list_ecosystem_modules(self):
        modules = self.get_list_all_pages(
            method="ecosystem.list",
            data={},
            convert_json_info_cb=lambda x: x,
        )
        if len(modules) == 0:
            raise KeyError("No modules found in ecosystem")
        return modules

    # def get_sessions(self, workspace_id: int, filter_statuses: List[TaskApi.Status] = None):
    #     filters = [{"field": "type", "operator": "=", "value": "app"}]
    #     # filters = []
    #     if filter_statuses is not None:
    #         s = [str(status) for status in filter_statuses]
    #         filters.append({"field": "status", "operator": "in", "value": s})
    #     result = self._api.task.get_list(workspace_id=workspace_id, filters=filters)
    #     return result

    def get_sessions(
        self,
        team_id,
        module_id,
        # only_running=False,
        show_disabled=False,
        session_name=None,
        statuses: List[TaskApi.Status] = None,
    ) -> List[SessionInfo]:
        infos_json = self.get_list_all_pages(
            method="apps.list",
            data={
                "teamId": team_id,
                "filter": [{"field": "moduleId", "operator": "=", "value": module_id}],
                # "onlyRunning": only_running,
                "showDisabled": show_disabled,
            },
            convert_json_info_cb=lambda x: x,
            # validate_total=False,
        )
        if len(infos_json) == 0:
            # raise KeyError(f"App [module_id = {module_id}] not found in team {team_id}")
            return []
        if len(infos_json) > 1:
            raise KeyError(
                f"Apps list in team is broken: app [module_id = {module_id}] added to team {team_id} multiple times"
            )
        dev_tasks = []
        sessions = infos_json[0]["tasks"]

        str_statuses = []
        if statuses is not None:
            for s in statuses:
                str_statuses.append(str(s))

        for session in sessions:
            to_add = True
            if session_name is not None and session["meta"]["name"] != session_name:
                to_add = False
            if statuses is not None and session["status"] not in str_statuses:
                to_add = False
            if to_add is True:
                session["moduleId"] = module_id
                dev_tasks.append(SessionInfo.from_json(session))
        return dev_tasks

    def start(
        self,
        agent_id,
        app_id=None,
        workspace_id=None,
        description="",
        params=None,
        log_level="info",
        users_id=None,
        app_version=None,
        is_branch=False,
        task_name="run-from-python",
        restart_policy="never",
        proxy_keep_url=False,
        module_id=None,
        redirect_requests={},
    ) -> SessionInfo:
        users_ids = None
        if users_id is not None:
            users_ids = [users_id]

        new_params = {}
        if "state" not in params:
            new_params["state"] = params
        else:
            new_params = params

        if app_version is None:
            module_info = self.get_ecosystem_module_info(module_id)
            app_version = module_info.get_latest_release().get("version", "")

        result = self._api.task.start(
            agent_id=agent_id,
            app_id=app_id,
            workspace_id=workspace_id,
            description=description,
            params=new_params,
            log_level=log_level,
            users_ids=users_ids,
            app_version=app_version,
            is_branch=is_branch,
            task_name=task_name,
            restart_policy=restart_policy,
            proxy_keep_url=proxy_keep_url,
            module_id=module_id,
            redirect_requests=redirect_requests,
        )
        if type(result) is not list:
            result = [result]
        if len(result) != 1:
            raise ValueError(f"{len(result)} tasks started instead of one")
        return SessionInfo.from_json(result[0])

    def wait(
        self,
        id: int,
        target_status: TaskApi.Status,
        attempts: Optional[int] = None,
        attempt_delay_sec: Optional[int] = None,
    ):
        """wait"""
        return self._api.task.wait(
            id=id,
            target_status=target_status,
            wait_attempts=attempts,
            wait_attempt_timeout_sec=attempt_delay_sec,
        )

    def stop(self, id: int) -> TaskApi.Status:
        """stop"""
        return self._api.task.stop(id)

    def get_status(self, task_id: int) -> TaskApi.Status:
        return self._api.task.get_status(task_id)


# info about app in team
# {
#     "id": 7,
#     "createdBy": 1,
#     "moduleId": 16,
#     "disabled": false,
#     "userLogin": "admin",
#     "config": {
#         "icon": "https://user-images.githubusercontent.com/12828725/182186256-5ee663ad-25c7-4a62-9af1-fbfdca715b57.png",
#         "author": {"name": "Maxim Kolomeychenko"},
#         "poster": "https://user-images.githubusercontent.com/12828725/182181033-d0d1a690-8388-472e-8862-e0cacbd4f082.png",
#         "needGPU": false,
#         "headless": true,
#         "categories": ["development"],
#         "lastCommit": "96eca85e1fbed45d59db405b17c04f4d920c6c81",
#         "description": "Demonstrates how to turn your python script into Supervisely App",
#         "main_script": "src/main.py",
#         "sessionTags": [],
#         "taskLocation": "workspace_tasks",
#         "defaultBranch": "master",
#         "isPrivateRepo": false,
#         "restartPolicy": "never",
#         "slyModuleInfo": {"baseSlug": "supervisely-ecosystem/hello-world-app"},
#         "communityAgent": true,
#         "iconBackground": "#FFFFFF",
#         "integratedInto": ["standalone"],
#         "storeDataOnAgent": false,
#     },
#     "name": "Hello World!",
#     "slug": "supervisely-ecosystem/hello-world-app",
#     "moduleDisabled": false,
#     "provider": "sly_gitea",
#     "repositoryId": 42,
#     "pathPrefix": "",
#     "baseUrl": null,
#     "isShared": false,
#     "tasks": [
#         {
#             "id": 19107,
#             "type": "app",
#             "size": "0",
#             "status": "finished",
#             "startedAt": "2022-08-04T14:59:45.797Z",
#             "finishedAt": "2022-08-04T14:59:49.793Z",
#             "meta": {
#                 "app": {
#                     "id": 7,
#                     "name": "Hello World!",
#                     "version": "v1.0.4",
#                     "isBranch": false,
#                     "logLevel": "info",
#                 },
#                 "name": "",
#                 "params": {"state": {}},
#                 "hasLogs": true,
#                 "logsCnt": 60,
#                 "hasMetrics": false,
#                 "sessionToken": "PDVBF6ecX09FY75n7ufa8q_MTB28XI6XIMcJ1md4ogeN0FLTbIZyC91Js_9YkGpUQhQbCYyTE8Q=",
#                 "restartPolicy": "never",
#             },
#             "attempt": 1,
#             "archived": false,
#             "nodeId": 1,
#             "createdBy": 6,
#             "teamId": 7,
#             "description": "",
#             "isShared": false,
#             "user": "max",
#         }
#     ],
#     "repo": "https://github.com/supervisely-ecosystem/hello-world-app",
#     "repoKey": "supervisely-ecosystem/hello-world-app",
#     "githubModuleUrl": "https://github.com/supervisely-ecosystem/hello-world-app",
#     "repositoryModuleUrl": "https://github.com/supervisely-ecosystem/hello-world-app",
#     "teamId": 7,
# }


# infor about module in ecosystem
