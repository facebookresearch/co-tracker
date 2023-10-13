# coding: utf-8
"""api for working with tasks"""

# docs
from typing import List, NamedTuple, Dict, Optional, Callable, Union
import os
import time
from collections import defaultdict, OrderedDict
import json
from tqdm import tqdm

from supervisely.api.module_api import (
    ApiField,
    ModuleApiBase,
    ModuleWithStatus,
    WaitingTimeExceeded,
)
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from supervisely.io.fs import get_file_name, ensure_base_path, get_file_hash
from supervisely.collection.str_enum import StrEnum
from supervisely._utils import batched, take_with_default


class TaskFinishedWithError(Exception):
    """TaskFinishedWithError"""

    pass


class TaskApi(ModuleApiBase, ModuleWithStatus):
    """
    API for working with Tasks. :class:`TaskApi<TaskApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
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

        task_id = 121230
        task_info = api.task.get_info_by_id(task_id)
    """

    class RestartPolicy(StrEnum):
        """RestartPolicy"""

        NEVER = "never"
        """"""
        ON_ERROR = "on_error"
        """"""

    class PluginTaskType(StrEnum):
        """PluginTaskType"""

        TRAIN = "train"
        """"""
        INFERENCE = "inference"
        """"""
        INFERENCE_RPC = "inference_rpc"
        """"""
        SMART_TOOL = "smarttool"
        """"""
        CUSTOM = "custom"
        """"""

    class Status(StrEnum):
        """Status"""

        QUEUED = "queued"
        """"""
        CONSUMED = "consumed"
        """"""
        STARTED = "started"
        """"""
        DEPLOYED = "deployed"
        """"""
        ERROR = "error"
        """"""
        FINISHED = "finished"
        """"""
        TERMINATING = "terminating"
        """"""
        STOPPED = "stopped"
        """"""

    def __init__(self, api):
        ModuleApiBase.__init__(self, api)
        ModuleWithStatus.__init__(self)

    def get_list(
        self, workspace_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[NamedTuple]:
        """
        List of Tasks in the given Workspace.

        :param workspace_id: Workspace ID.
        :type workspace_id: int
        :param filters: List of params to sort output Projects.
        :type filters: List[dict], optional
        :return: List of Tasks with information for the given Workspace.
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            workspace_id = 23821

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            task_infos = api.task.get_list(workspace_id)

            task_infos_filter = api.task.get_list(23821, filters=[{'field': 'id', 'operator': '=', 'value': 121230}])
            print(task_infos_filter)
            # Output: [
            #     {
            #         "id": 121230,
            #         "type": "clone",
            #         "status": "finished",
            #         "startedAt": "2019-12-19T12:13:09.702Z",
            #         "finishedAt": "2019-12-19T12:13:09.701Z",
            #         "meta": {
            #             "input": {
            #                 "model": {
            #                     "id": 1849
            #                 },
            #                 "isExternal": true,
            #                 "pluginVersionId": 84479
            #             },
            #             "output": {
            #                 "model": {
            #                     "id": 12380
            #                 },
            #                 "pluginVersionId": 84479
            #             }
            #         },
            #         "description": ""
            #     }
            # ]
        """
        return self.get_list_all_pages(
            "tasks.list",
            {ApiField.WORKSPACE_ID: workspace_id, ApiField.FILTER: filters or []},
        )

    def get_info_by_id(self, id: int) -> NamedTuple:
        """
        Get Task information by ID.

        :param id: Task ID in Supervisely.
        :type id: int
        :return: Information about Task.
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            task_id = 121230

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            task_info = api.task.get_info_by_id(task_id)
            print(task_info)
            # Output: {
            #     "id": 121230,
            #     "workspaceId": 23821,
            #     "description": "",
            #     "type": "clone",
            #     "status": "finished",
            #     "startedAt": "2019-12-19T12:13:09.702Z",
            #     "finishedAt": "2019-12-19T12:13:09.701Z",
            #     "userId": 16154,
            #     "meta": {
            #         "input": {
            #             "model": {
            #                 "id": 1849
            #             },
            #             "isExternal": true,
            #             "pluginVersionId": 84479
            #         },
            #         "output": {
            #             "model": {
            #                 "id": 12380
            #             },
            #             "pluginVersionId": 84479
            #         }
            #     },
            #     "settings": {},
            #     "agentName": null,
            #     "userLogin": "alexxx",
            #     "teamId": 16087,
            #     "agentId": null
            # }
        """
        return self._get_info_by_id(id, "tasks.info")

    def get_status(self, task_id: int) -> Status:
        """
        Check status of Task by ID.

        :param id: Task ID in Supervisely.
        :type id: int
        :return: Status object
        :rtype: :class:`Status`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            task_id = 121230

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            task_status = api.task.get_status(task_id)
            print(task_status)
            # Output: finished
        """
        status_str = self.get_info_by_id(task_id)[ApiField.STATUS]  # @TODO: convert json to tuple
        return self.Status(status_str)

    def raise_for_status(self, status: Status) -> None:
        """
        Raise error if Task status is ERROR.

        :param status: Status object.
        :type status: Status
        :return: None
        :rtype: :class:`NoneType`
        """
        if status is self.Status.ERROR:
            raise TaskFinishedWithError(f"Task finished with status {str(self.Status.ERROR)}")

    def wait(
        self,
        id: int,
        target_status: Status,
        wait_attempts: Optional[int] = None,
        wait_attempt_timeout_sec: Optional[int] = None,
    ):
        """
        Awaiting achievement by given Task of a given status.

        :param id: Task ID in Supervisely.
        :type id: int
        :param target_status: Status object(status of task we expect to destinate).
        :type target_status: Status
        :param wait_attempts: The number of attempts to determine the status of the task that we are waiting for.
        :type wait_attempts: int, optional
        :param wait_attempt_timeout_sec: Number of seconds for intervals between attempts(raise error if waiting time exceeded).
        :type wait_attempt_timeout_sec: int, optional
        :return: True if the desired status is reached, False otherwise
        :rtype: :class:`bool`
        """
        wait_attempts = wait_attempts or self.MAX_WAIT_ATTEMPTS
        effective_wait_timeout = wait_attempt_timeout_sec or self.WAIT_ATTEMPT_TIMEOUT_SEC
        for attempt in range(wait_attempts):
            status = self.get_status(id)
            self.raise_for_status(status)
            if status in [
                target_status,
                self.Status.FINISHED,
                self.Status.DEPLOYED,
                self.Status.STOPPED,
            ]:
                return
            time.sleep(effective_wait_timeout)
        raise WaitingTimeExceeded(
            f"Waiting time exceeded: total waiting time {wait_attempts * effective_wait_timeout} seconds, i.e. {wait_attempts} attempts for {effective_wait_timeout} seconds each"
        )

    def upload_dtl_archive(
        self, task_id: int, archive_path: str, progress_cb: Optional[Union[tqdm, Callable]] = None
    ):
        """upload_dtl_archive"""
        encoder = MultipartEncoder(
            {
                "id": str(task_id).encode("utf-8"),
                "name": get_file_name(archive_path),
                "archive": (
                    os.path.basename(archive_path),
                    open(archive_path, "rb"),
                    "application/x-tar",
                ),
            }
        )

        def callback(monitor_instance):
            read_mb = monitor_instance.bytes_read / 1024.0 / 1024.0
            if progress_cb is not None:
                progress_cb(read_mb)

        monitor = MultipartEncoderMonitor(encoder, callback)
        self._api.post("tasks.upload.dtl_archive", monitor)

    def _deploy_model(
        self,
        agent_id,
        model_id,
        plugin_id=None,
        version=None,
        restart_policy=RestartPolicy.NEVER,
        settings=None,
    ):
        """_deploy_model"""
        response = self._api.post(
            "tasks.run.deploy",
            {
                ApiField.AGENT_ID: agent_id,
                ApiField.MODEL_ID: model_id,
                ApiField.RESTART_POLICY: restart_policy.value,
                ApiField.SETTINGS: settings or {"gpu_device": 0},
                ApiField.PLUGIN_ID: plugin_id,
                ApiField.VERSION: version,
            },
        )
        return response.json()[ApiField.TASK_ID]

    def get_context(self, id: int) -> Dict:
        """
        Get context information by task ID.

        :param id: Task ID in Supervisely.
        :type id: int
        :return: Context information in dict format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            task_id = 121230

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            context = api.task.get_context(task_id)
            print(context)
            # Output: {
            #     "team": {
            #         "id": 16087,
            #         "name": "alexxx"
            #     },
            #     "workspace": {
            #         "id": 23821,
            #         "name": "my_super_workspace"
            #     }
            # }
        """
        response = self._api.post("GetTaskContext", {ApiField.ID: id})
        return response.json()

    def _convert_json_info(self, info: dict):
        """_convert_json_info"""
        return info

    def run_dtl(self, workspace_id: int, dtl_graph: Dict, agent_id: Optional[int] = None):
        """run_dtl"""
        response = self._api.post(
            "tasks.run.dtl",
            {
                ApiField.WORKSPACE_ID: workspace_id,
                ApiField.CONFIG: dtl_graph,
                "advanced": {ApiField.AGENT_ID: agent_id},
            },
        )
        return response.json()[ApiField.TASK_ID]

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
        response = self._api.post(
            "tasks.run.plugin",
            {
                "taskType": task_type,
                ApiField.AGENT_ID: agent_id,
                ApiField.PLUGIN_ID: plugin_id,
                ApiField.VERSION: version,
                ApiField.CONFIG: config,
                "projects": input_projects,
                "models": input_models,
                ApiField.NAME: result_name,
            },
        )
        return response.json()[ApiField.TASK_ID]

    def run_train(
        self,
        agent_id: int,
        input_project_id: int,
        input_model_id: int,
        result_nn_name: str,
        train_config: Optional[Dict] = None,
    ):
        """run_train"""
        model_info = self._api.model.get_info_by_id(input_model_id)
        return self._run_plugin_task(
            task_type=TaskApi.PluginTaskType.TRAIN.value,
            agent_id=agent_id,
            plugin_id=model_info.plugin_id,
            version=None,
            input_projects=[input_project_id],
            input_models=[input_model_id],
            result_name=result_nn_name,
            config={} if train_config is None else train_config,
        )

    def run_inference(
        self,
        agent_id: int,
        input_project_id: int,
        input_model_id: int,
        result_project_name: str,
        inference_config: Optional[Dict] = None,
    ):
        """run_inference"""
        model_info = self._api.model.get_info_by_id(input_model_id)
        return self._run_plugin_task(
            task_type=TaskApi.PluginTaskType.INFERENCE.value,
            agent_id=agent_id,
            plugin_id=model_info.plugin_id,
            version=None,
            input_projects=[input_project_id],
            input_models=[input_model_id],
            result_name=result_project_name,
            config={} if inference_config is None else inference_config,
        )

    def get_training_metrics(self, task_id: int):
        """get_training_metrics"""
        response = self._get_response_by_id(
            id=task_id, method="tasks.train-metrics", id_field=ApiField.TASK_ID
        )
        return response.json() if (response is not None) else None

    def deploy_model(self, agent_id: int, model_id: int) -> int:
        """deploy_model"""
        task_ids = self._api.model.get_deploy_tasks(model_id)
        if len(task_ids) == 0:
            task_id = self._deploy_model(agent_id, model_id)
        else:
            task_id = task_ids[0]
        self.wait(task_id, self.Status.DEPLOYED)
        return task_id

    def deploy_model_async(self, agent_id: int, model_id: int) -> int:
        """deploy_model_async"""
        task_ids = self._api.model.get_deploy_tasks(model_id)
        if len(task_ids) == 0:
            task_id = self._deploy_model(agent_id, model_id)
        else:
            task_id = task_ids[0]
        return task_id

    def start(
        self,
        agent_id,
        app_id=None,
        workspace_id=None,
        description="application description",
        params=None,
        log_level="info",
        users_ids=None,
        app_version="",
        is_branch=False,
        task_name="pythonSpawned",
        restart_policy="never",
        proxy_keep_url=False,
        module_id=None,
        redirect_requests={},
    ):
        """start"""
        if app_id is not None and module_id is not None:
            raise ValueError("Only one of the arguments (app_id or module_id) have to be defined")
        if app_id is None and module_id is None:
            raise ValueError("One of the arguments (app_id or module_id) have to be defined")

        data = {
            ApiField.AGENT_ID: agent_id,
            # "nodeId": agent_id,
            ApiField.WORKSPACE_ID: workspace_id,
            ApiField.DESCRIPTION: description,
            ApiField.PARAMS: take_with_default(params, {"state": {}}),
            ApiField.LOG_LEVEL: log_level,
            ApiField.USERS_IDS: take_with_default(users_ids, []),
            ApiField.APP_VERSION: app_version,
            ApiField.IS_BRANCH: is_branch,
            ApiField.TASK_NAME: task_name,
            ApiField.RESTART_POLICY: restart_policy,
            ApiField.PROXY_KEEP_URL: proxy_keep_url,
        }
        if len(redirect_requests) > 0:
            data[ApiField.REDIRECT_REQUESTS] = redirect_requests

        if app_id is not None:
            data[ApiField.APP_ID] = app_id
        if module_id is not None:
            data[ApiField.MODULE_ID] = module_id
        resp = self._api.post(method="tasks.run.app", data=data)
        task = resp.json()[0]
        if "id" not in task:
            task["id"] = task.get("taskId")
        return task

    def stop(self, id: int):
        """stop"""
        response = self._api.post("tasks.stop", {ApiField.ID: id})
        return self.Status(response.json()[ApiField.STATUS])

    def get_import_files_list(self, id: int) -> Dict or None:
        """get_import_files_list"""
        response = self._api.post("tasks.import.files_list", {ApiField.ID: id})
        return response.json() if (response is not None) else None

    def download_import_file(self, id, file_path, save_path):
        """download_import_file"""
        response = self._api.post(
            "tasks.import.download_file",
            {ApiField.ID: id, ApiField.FILENAME: file_path},
            stream=True,
        )

        ensure_base_path(save_path)
        with open(save_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def create_task_detached(self, workspace_id: int, task_type: Optional[str] = None):
        """create_task_detached"""
        response = self._api.post(
            "tasks.run.python",
            {
                ApiField.WORKSPACE_ID: workspace_id,
                ApiField.SCRIPT: "xxx",
                ApiField.ADVANCED: {ApiField.IGNORE_AGENT: True},
            },
        )
        return response.json()[ApiField.TASK_ID]

    def submit_logs(self, logs) -> None:
        """submit_logs"""
        response = self._api.post("tasks.logs.add", {ApiField.LOGS: logs})
        # return response.json()[ApiField.TASK_ID]

    def upload_files(
        self,
        task_id: int,
        abs_paths: List[str],
        names: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """upload_files"""
        if len(abs_paths) != len(names):
            raise RuntimeError("Inconsistency: len(abs_paths) != len(names)")

        hashes = []
        if len(abs_paths) == 0:
            return

        hash_to_items = defaultdict(list)
        hash_to_name = defaultdict(list)
        for idx, item in enumerate(zip(abs_paths, names)):
            path, name = item
            item_hash = get_file_hash(path)
            hashes.append(item_hash)
            hash_to_items[item_hash].append(path)
            hash_to_name[item_hash].append(name)

        unique_hashes = set(hashes)
        remote_hashes = self._api.image.check_existing_hashes(list(unique_hashes))
        new_hashes = unique_hashes - set(remote_hashes)

        # @TODO: upload remote hashes
        if len(remote_hashes) != 0:
            files = []
            for hash in remote_hashes:
                for name in hash_to_name[hash]:
                    files.append({ApiField.NAME: name, ApiField.HASH: hash})
            for batch in batched(files):
                resp = self._api.post(
                    "tasks.files.bulk.add-by-hash",
                    {ApiField.TASK_ID: task_id, ApiField.FILES: batch},
                )
        if progress_cb is not None:
            progress_cb(len(remote_hashes))

        for batch in batched(list(zip(abs_paths, names, hashes))):
            content_dict = OrderedDict()
            for idx, item in enumerate(batch):
                path, name, hash = item
                if hash in remote_hashes:
                    continue
                content_dict["{}".format(idx)] = json.dumps({"fullpath": name, "hash": hash})
                content_dict["{}-file".format(idx)] = (name, open(path, "rb"), "")

            if len(content_dict) > 0:
                encoder = MultipartEncoder(fields=content_dict)
                resp = self._api.post("tasks.files.bulk.upload", encoder)
                if progress_cb is not None:
                    progress_cb(len(content_dict))

    # {
    #     data: {my_val: 1}
    #     obj: {val: 1, res: 2}
    # }
    # {
    #     obj: {new_val: 1}
    # }
    # // apped: true, recursive: false
    # {
    #     data: {my_val: 1}
    #     obj: {new_val: 1}
    # }(edited)
    # // append: false, recursive: false
    # {
    #     obj: {new_val: 1}
    # }(edited)
    #
    # 16: 32
    # // append: true, recursive: true
    # {
    #     data: {my_val: 1}
    #     obj: {val: 1, res: 2, new_val: 1}
    # }

    def set_fields(self, task_id: int, fields: List) -> Dict:
        """set_fields"""
        for idx, obj in enumerate(fields):
            for key in [ApiField.FIELD, ApiField.PAYLOAD]:
                if key not in obj:
                    raise KeyError("Object #{} does not have field {!r}".format(idx, key))
        data = {ApiField.TASK_ID: task_id, ApiField.FIELDS: fields}
        resp = self._api.post("tasks.data.set", data)
        return resp.json()

    def set_fields_from_dict(self, task_id: int, d: Dict) -> Dict:
        """set_fields_from_dict"""
        fields = []
        for k, v in d.items():
            fields.append({ApiField.FIELD: k, ApiField.PAYLOAD: v})
        return self.set_fields(task_id, fields)

    def set_field(
        self,
        task_id: int,
        field: Dict,
        payload: Dict,
        append: Optional[bool] = False,
        recursive: Optional[bool] = False,
    ) -> Dict:
        """set_field"""
        fields = [
            {
                ApiField.FIELD: field,
                ApiField.PAYLOAD: payload,
                ApiField.APPEND: append,
                ApiField.RECURSIVE: recursive,
            }
        ]
        return self.set_fields(task_id, fields)

    def get_fields(self, task_id, fields: List):
        """get_fields"""
        data = {ApiField.TASK_ID: task_id, ApiField.FIELDS: fields}
        resp = self._api.post("tasks.data.get", data)
        return resp.json()["result"]

    def get_field(self, task_id: int, field: Dict):
        """get_field"""
        result = self.get_fields(task_id, [field])
        return result[field]

    def _validate_checkpoints_support(self, task_id):
        """_validate_checkpoints_support"""
        info = self.get_info_by_id(task_id)
        if info["type"] != str(TaskApi.PluginTaskType.TRAIN):
            raise RuntimeError(
                "Task (id={!r}) has type {!r}. "
                "Checkpoints are available only for tasks of type {!r}".format()
            )

    def list_checkpoints(self, task_id: int):
        """list_checkpoints"""
        self._validate_checkpoints_support(task_id)
        resp = self._api.post("tasks.checkpoints.list", {ApiField.ID: task_id})
        return resp.json()

    def delete_unused_checkpoints(self, task_id: int) -> Dict:
        """delete_unused_checkpoints"""
        self._validate_checkpoints_support(task_id)
        resp = self._api.post("tasks.checkpoints.clear", {ApiField.ID: task_id})
        return resp.json()

    def _set_output(self):
        """_set_output"""
        pass

    def set_output_project(
        self, task_id: int, project_id: int, project_name: Optional[str] = None
    ) -> Dict:
        """set_output_project"""
        if project_name is None:
            project = self._api.project.get_info_by_id(project_id, raise_error=True)
            project_name = project.name

        output = {ApiField.PROJECT: {ApiField.ID: project_id, ApiField.TITLE: project_name}}
        resp = self._api.post(
            "tasks.output.set", {ApiField.TASK_ID: task_id, ApiField.OUTPUT: output}
        )
        return resp.json()

    def set_output_report(self, task_id: int, file_id: int, file_name: str) -> Dict:
        """set_output_report"""
        return self._set_custom_output(
            task_id, file_id, file_name, description="Report", icon="zmdi zmdi-receipt"
        )

    def _set_custom_output(
        self,
        task_id,
        file_id,
        file_name,
        file_url=None,
        description="File",
        icon="zmdi zmdi-file-text",
        color="#33c94c",
        background_color="#d9f7e4",
        download=False,
    ):
        """_set_custom_output"""
        if file_url is None:
            file_url = self._api.file.get_url(file_id)

        output = {
            ApiField.GENERAL: {
                "icon": {
                    "className": icon,
                    "color": color,
                    "backgroundColor": background_color,
                },
                "title": file_name,
                "titleUrl": file_url,
                "download": download,
                "description": description,
            }
        }
        resp = self._api.post(
            "tasks.output.set", {ApiField.TASK_ID: task_id, ApiField.OUTPUT: output}
        )
        return resp.json()

    def set_output_archive(
        self, task_id: int, file_id: int, file_name: str, file_url: Optional[str] = None
    ) -> Dict:
        """set_output_archive"""
        if file_url is None:
            file_url = self._api.file.get_info_by_id(file_id).storage_path
        return self._set_custom_output(
            task_id,
            file_id,
            file_name,
            file_url=file_url,
            description="Download archive",
            icon="zmdi zmdi-archive",
            download=True,
        )

    def set_output_file_download(
        self,
        task_id: int,
        file_id: int,
        file_name: str,
        file_url: Optional[str] = None,
        download: Optional[bool] = True,
    ) -> Dict:
        """set_output_file_download"""
        if file_url is None:
            file_url = self._api.file.get_info_by_id(file_id).storage_path
        return self._set_custom_output(
            task_id,
            file_id,
            file_name,
            file_url=file_url,
            description="Download file",
            icon="zmdi zmdi-file",
            download=download,
        )

    def send_request(
        self,
        task_id: int,
        method: str,
        data: Dict,
        context: Optional[Dict] = {},
        skip_response: bool = False,
        timeout: Optional[int] = 60,
        outside_request: bool = True,
    ):
        """send_request"""
        if type(data) is not dict:
            raise TypeError("data argument has to be a dict")
        context["outside_request"] = outside_request
        resp = self._api.post(
            "tasks.request.direct",
            {
                ApiField.TASK_ID: task_id,
                ApiField.COMMAND: method,
                ApiField.CONTEXT: context,
                ApiField.STATE: data,
                "skipResponse": skip_response,
                "timeout": timeout,
            },
        )
        return resp.json()

    def set_output_directory(self, task_id, file_id, directory_path):
        """set_output_directory"""
        return self._set_custom_output(
            task_id,
            file_id,
            directory_path,
            description="Directory",
            icon="zmdi zmdi-folder",
        )

    def update_meta(
        self, id: int, data: dict, agent_storage_folder: str = None, relative_app_dir: str = None
    ):
        """
        Update given task metadata
        :param id: int — task id
        :param data: dict — meta data to update
        """
        if type(data) == dict:
            data.update({"id": id})
            if agent_storage_folder is None and relative_app_dir is not None:
                raise ValueError(
                    "Both arguments (agent_storage_folder and relative_app_dir) has to be defined or None"
                )
            if agent_storage_folder is not None and relative_app_dir is None:
                raise ValueError(
                    "Both arguments (agent_storage_folder and relative_app_dir) has to be defined or None"
                )
            if agent_storage_folder is not None and relative_app_dir is not None:
                data["agentStorageFolder"] = {
                    "hostDir": agent_storage_folder,
                    "folder": relative_app_dir,
                }

        self._api.post("tasks.meta.update", data)

    def _update_app_content(self, task_id: int, data_patch: List[Dict] = None, state: Dict = None):
        payload = {}
        if data_patch is not None and len(data_patch) > 0:
            payload[ApiField.DATA] = data_patch
        if state is not None and len(state) > 0:
            payload[ApiField.STATE] = state

        resp = self._api.post(
            "tasks.app-v2.data.set",
            {ApiField.TASK_ID: task_id, ApiField.PAYLOAD: payload},
        )
        return resp.json()
