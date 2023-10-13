# coding: utf-8
"""api connection to the server which allows user to communicate with Supervisely"""

from __future__ import annotations
import os
import requests
import json
from typing import List, Optional, NamedTuple, Dict
from requests_toolbelt import MultipartEncoderMonitor, MultipartEncoder
from dotenv import load_dotenv


import supervisely.api.team_api as team_api
import supervisely.api.workspace_api as workspace_api
import supervisely.api.project_api as project_api
import supervisely.api.neural_network_api as neural_network_api
import supervisely.api.task_api as task_api
import supervisely.api.dataset_api as dataset_api
import supervisely.api.image_api as image_api
import supervisely.api.annotation_api as annotation_api
import supervisely.api.plugin_api as plugin_api
import supervisely.api.agent_api as agent_api
import supervisely.api.role_api as role_api
import supervisely.api.user_api as user_api
import supervisely.api.labeling_job_api as labeling_job_api
import supervisely.api.video.video_api as video_api
import supervisely.api.pointcloud.pointcloud_api as pointcloud_api
import supervisely.api.pointcloud.pointcloud_episode_api as pointcloud_episode_api
import supervisely.api.object_class_api as object_class_api
import supervisely.api.report_api as report_api
import supervisely.api.app_api as app_api
import supervisely.api.file_api as file_api
import supervisely.api.image_annotation_tool_api as image_annotation_tool_api
import supervisely.api.advanced_api as advanced_api
import supervisely.api.import_storage_api as import_stoarge_api
import supervisely.api.remote_storage_api as remote_storage_api
import supervisely.api.github_api as github_api
import supervisely.api.volume.volume_api as volume_api
from supervisely.sly_logger import logger
import supervisely.io.env as sly_env
from supervisely._utils import is_development


from supervisely.io.network_exceptions import (
    process_requests_exception,
    process_unhandled_request,
)

SUPERVISELY_TASK_ID = "SUPERVISELY_TASK_ID"
SUPERVISELY_PUBLIC_API_RETRIES = "SUPERVISELY_PUBLIC_API_RETRIES"
SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC = "SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC"
SERVER_ADDRESS = "SERVER_ADDRESS"
API_TOKEN = "API_TOKEN"
TASK_ID = "TASK_ID"


class Api:
    """
    An API connection to the server with which you can communicate with your teams, workspaces and projects. :class:`Api<Api>` object is immutable.

    :param server_address: Address of the server.
    :type server_address: str
    :param token: Unique secret token associated with your agent.
    :type token: str
    :param retry_count: The number of attempts to connect to the server.
    :type retry_count: int, optional
    :param retry_sleep_sec: The number of seconds to delay between attempts to connect to the server.
    :type retry_sleep_sec: int, optional
    :param external_logger: Logger class object.
    :type external_logger: logger, optional
    :param ignore_task_id:
    :type ignore_task_id: bool, optional
    :raises: :class:`ValueError`, if token is None or it length != 128
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
    """

    def __init__(
        self,
        server_address: str = None,
        token: str = None,
        retry_count: Optional[int] = 10,
        retry_sleep_sec: Optional[int] = None,
        external_logger: Optional[logger] = None,
        ignore_task_id: Optional[bool] = False,
    ):
        if server_address is None and token is None:
            server_address = os.environ.get(SERVER_ADDRESS, None)
            token = os.environ.get(API_TOKEN, None)

        if server_address is None:
            raise ValueError(
                "SERVER_ADDRESS env variable is undefined, https://developer.supervise.ly/getting-started/basics-of-authentication"
            )
        if token is None:
            raise ValueError(
                "API_TOKEN env variable is undefined, https://developer.supervise.ly/getting-started/basics-of-authentication"
            )
        self.server_address = Api.normalize_server_address(server_address)

        if retry_count is None:
            retry_count = int(os.getenv(SUPERVISELY_PUBLIC_API_RETRIES, "10"))
        if retry_sleep_sec is None:
            retry_sleep_sec = int(os.getenv(SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC, "1"))

        if len(token) != 128:
            raise ValueError("Invalid token {!r}: length != 128".format(token))

        self.token = token
        self.headers = {"x-api-key": token}
        self.task_id = os.getenv(SUPERVISELY_TASK_ID)
        if self.task_id is not None and ignore_task_id is False:
            self.headers["x-task-id"] = self.task_id
        self.context = {}
        self.additional_fields = {}

        self.team = team_api.TeamApi(self)
        self.workspace = workspace_api.WorkspaceApi(self)
        self.project = project_api.ProjectApi(self)
        self.model = neural_network_api.NeuralNetworkApi(self)
        self.task = task_api.TaskApi(self)
        self.dataset = dataset_api.DatasetApi(self)
        self.image = image_api.ImageApi(self)
        self.annotation = annotation_api.AnnotationApi(self)
        self.plugin = plugin_api.PluginApi(self)
        self.agent = agent_api.AgentApi(self)
        self.role = role_api.RoleApi(self)
        self.user = user_api.UserApi(self)
        self.labeling_job = labeling_job_api.LabelingJobApi(self)
        self.video = video_api.VideoApi(self)
        # self.project_class = project_class_api.ProjectClassApi(self)
        self.object_class = object_class_api.ObjectClassApi(self)
        self.report = report_api.ReportApi(self)
        self.pointcloud = pointcloud_api.PointcloudApi(self)
        self.pointcloud_episode = pointcloud_episode_api.PointcloudEpisodeApi(self)
        self.app = app_api.AppApi(self)
        self.file = file_api.FileApi(self)
        self.img_ann_tool = image_annotation_tool_api.ImageAnnotationToolApi(self)
        self.advanced = advanced_api.AdvancedApi(self)
        self.import_storage = import_stoarge_api.ImportStorageApi(self)
        self.remote_storage = remote_storage_api.RemoteStorageApi(self)
        self.github = github_api.GithubApi(self)
        self.volume = volume_api.VolumeApi(self)

        self.retry_count = retry_count
        self.retry_sleep_sec = retry_sleep_sec

        self.logger = external_logger or logger

    @classmethod
    def normalize_server_address(cls, server_address):
        """ """
        result = server_address.strip("/")
        if ("http://" not in result) and ("https://" not in result):
            result = "http://" + result
        return result

    @classmethod
    def from_env(
        cls,
        retry_count: int = 10,
        ignore_task_id: bool = False,
        env_file: str = "~/supervisely.env",
    ) -> Api:
        """
        Initialize API use environment variables.

        :param retry_count: The number of attempts to connect to the server.
        :type retry_count: int
        :param ignore_task_id:
        :type ignore_task_id: bool
        :param path: Path to your .env file.
        :type path: str
        :return: Api object
        :rtype: :class:`Api<supervisely.api.api.Api>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'

            api = sly.Api.from_env()

            # alternatively you can store SERVER_ADDRESS and API_TOKEN
            # in "~/supervisely.env" .env file
            # Learn more here: https://developer.supervise.ly/app-development/basics/add-private-app#create-.env-file-supervisely.env-with-the-following-content-learn-more-here

            api = sly.Api.from_env()
        """

        server_address = sly_env.server_address(raise_not_found=False)
        token = sly_env.api_token(raise_not_found=False)

        if is_development() and None in (server_address, token):
            env_path = os.path.expanduser(env_file)
            if os.path.exists(env_path):
                _, extension = os.path.splitext(env_path)
                if extension == ".env":
                    load_dotenv(env_path)
                    server_address = sly_env.server_address()
                    token = sly_env.api_token()
                else:
                    raise ValueError(f"'{env_path}' is not an '*.env' file")
            else:
                raise FileNotFoundError(f"File not found: '{env_path}'")

        if server_address is None:
            raise ValueError(
                "SERVER_ADDRESS env variable is undefined. Learn more here: https://developer.supervise.ly/getting-started/basics-of-authentication"
            )
        if token is None:
            raise ValueError(
                "API_TOKEN env variable is undefined. Learn more here: https://developer.supervise.ly/getting-started/basics-of-authentication"
            )

        return cls(
            server_address,
            token,
            retry_count=retry_count,
            ignore_task_id=ignore_task_id,
        )

    def add_header(self, key: str, value: str) -> None:
        """
        Add given key and value to headers dictionary.

        :param key: New key.
        :type key: str
        :param value: New value.
        :type value: str
        :raises: :class:`RuntimeError`, if key is already set
        :return: None
        :rtype: :class:`NoneType`
        """
        if key in self.headers:
            raise RuntimeError(
                f"Header {key!r} is already set for the API object. "
                f"Current value: {self.headers[key]!r}. Tried to set value: {value!r}"
            )
        self.headers[key] = value

    def add_additional_field(self, key: str, value: str) -> None:
        """
        Add given key and value to additional_fields dictionary.

        :param key: New key.
        :type key: str
        :param value: New value.
        :type value: str
        :return: None
        :rtype: :class:`NoneType`
        """
        self.additional_fields[key] = value

    def post(
        self,
        method: str,
        data: Dict,
        retries: Optional[int] = None,
        stream: Optional[bool] = False,
    ) -> requests.Response:
        """
        Performs POST request to server with given parameters.

        :param method:
        :type method: str
        :param data: Dictionary to send in the body of the :class:`Request`.
        :type data: dict
        :param retries: The number of attempts to connect to the server.
        :type retries: int, optional
        :param stream: Define, if you’d like to get the raw socket response from the server.
        :type stream: bool, optional
        :return: Response object
        :rtype: :class:`Response<Response>`
        """
        if retries is None:
            retries = self.retry_count

        url = self.server_address + "/public/api/v3/" + method
        logger.trace(f"POST {url}")

        for retry_idx in range(retries):
            response = None
            try:
                if type(data) is bytes:
                    response = requests.post(url, data=data, headers=self.headers, stream=stream)
                elif type(data) is MultipartEncoderMonitor or type(data) is MultipartEncoder:
                    response = requests.post(
                        url,
                        data=data,
                        headers={**self.headers, "Content-Type": data.content_type},
                        stream=stream,
                    )
                else:
                    json_body = data
                    if type(data) is dict:
                        json_body = {**data, **self.additional_fields}
                    response = requests.post(
                        url, json=json_body, headers=self.headers, stream=stream
                    )

                if response.status_code != requests.codes.ok:
                    Api._raise_for_status(response)
                return response
            except requests.RequestException as exc:
                process_requests_exception(
                    self.logger,
                    exc,
                    method,
                    url,
                    verbose=True,
                    swallow_exc=True,
                    sleep_sec=min(self.retry_sleep_sec * (2**retry_idx), 60),
                    response=response,
                    retry_info={"retry_idx": retry_idx + 1, "retry_limit": retries},
                )
            except Exception as exc:
                process_unhandled_request(self.logger, exc)
        raise requests.exceptions.RetryError("Retry limit exceeded ({!r})".format(url))

    def get(
        self,
        method: str,
        params: Dict,
        retries: Optional[int] = None,
        stream: Optional[bool] = False,
        use_public_api: Optional[bool] = True,
    ) -> requests.Response:
        """
        Performs GET request to server with given parameters.

        :param method:
        :type method: str
        :param params: Dictionary to send in the body of the :class:`Request`.
        :type method: dict
        :param retries: The number of attempts to connect to the server.
        :type method: int, optional
        :param stream: Define, if you’d like to get the raw socket response from the server.
        :type method: bool, optional
        :param use_public_api:
        :type method: bool, optional
        :return: Response object
        :rtype: :class:`Response<Response>`
        """
        if retries is None:
            retries = self.retry_count

        url = self.server_address + "/public/api/v3/" + method
        if use_public_api is False:
            url = os.path.join(self.server_address, method)
        logger.trace(f"GET {url}")

        for retry_idx in range(retries):
            response = None
            try:
                json_body = params
                if type(params) is dict:
                    json_body = {**params, **self.additional_fields}
                response = requests.get(url, params=json_body, headers=self.headers, stream=stream)

                if response.status_code != requests.codes.ok:
                    Api._raise_for_status(response)
                return response
            except requests.RequestException as exc:
                process_requests_exception(
                    self.logger,
                    exc,
                    method,
                    url,
                    verbose=True,
                    swallow_exc=True,
                    sleep_sec=min(self.retry_sleep_sec * (2**retry_idx), 60),
                    response=response,
                    retry_info={"retry_idx": retry_idx + 2, "retry_limit": retries},
                )
            except Exception as exc:
                process_unhandled_request(self.logger, exc)

    @staticmethod
    def _raise_for_status(response):
        """
        Raise error and show message with error code if given response can not connect to server.
        :param response: Request class object
        """
        http_error_msg = ""
        if isinstance(response.reason, bytes):
            try:
                reason = response.reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = response.reason.decode("iso-8859-1")
        else:
            reason = response.reason

        if 400 <= response.status_code < 500:
            http_error_msg = "%s Client Error: %s for url: %s (%s)" % (
                response.status_code,
                reason,
                response.url,
                response.content.decode("utf-8"),
            )

        elif 500 <= response.status_code < 600:
            http_error_msg = "%s Server Error: %s for url: %s (%s)" % (
                response.status_code,
                reason,
                response.url,
                response.content.decode("utf-8"),
            )

        if http_error_msg:
            raise requests.exceptions.HTTPError(http_error_msg, response=response)

    @staticmethod
    def parse_error(
        response: requests.Response,
        default_error: Optional[str] = "Error",
        default_message: Optional[str] = "please, contact administrator",
    ):
        """
        Processes error from response.

        :param response: Request object.
        :type method: Request
        :param default_error: Error description.
        :type method: str, optional
        :param default_message: Message to user.
        :type method: str, optional
        :return: Number of error and message about curren connection mistake
        :rtype: :class:`int`, :class:`str`
        """
        ERROR_FIELD = "error"
        MESSAGE_FIELD = "message"
        DETAILS_FIELD = "details"

        try:
            data_str = response.content.decode("utf-8")
            data = json.loads(data_str)
            error = data.get(ERROR_FIELD, default_error)
            details = data.get(DETAILS_FIELD, {})
            if type(details) is dict:
                message = details.get(MESSAGE_FIELD, default_message)
            else:
                message = details[0].get(MESSAGE_FIELD, default_message)

            return error, message
        except Exception as e:
            return "", ""

    def pop_header(self, key: str) -> str:
        """ """
        if key not in self.headers:
            raise KeyError(f"Header {key!r} not found")
        return self.headers.pop(key)
