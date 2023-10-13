# coding: utf-8
"""list/create teams and monitor their activity"""

# docs
from __future__ import annotations
from typing import NamedTuple, List, Dict, Optional, Callable, Union
from supervisely.task.progress import Progress

from supervisely.api.module_api import ApiField, ModuleNoParent, UpdateableModule
from supervisely.sly_logger import logger
from tqdm import tqdm


# @TODO - umar will add meta with review status and duration
class ActivityAction:
    """
    List of Team Actions to sort Team Activity.
    """

    LOGIN = "login"
    """"""
    LOGOUT = "logout"
    """"""
    CREATE_PROJECT = "create_project"
    """"""
    UPDATE_PROJECT = "update_project"
    """"""
    DISABLE_PROJECT = "disable_project"
    """"""
    RESTORE_PROJECT = "restore_project"
    """"""
    CREATE_DATASET = "create_dataset"
    """"""
    UPDATE_DATASET = "update_dataset"
    """"""
    DISABLE_DATASET = "disable_dataset"
    """"""
    RESTORE_DATASET = "restore_dataset"
    """"""
    CREATE_IMAGE = "create_image"
    """"""
    UPDATE_IMAGE = "update_image"
    """"""
    DISABLE_IMAGE = "disable_image"
    """"""
    RESTORE_IMAGE = "restore_image"
    """"""
    CREATE_FIGURE = "create_figure"
    """"""
    UPDATE_FIGURE = "update_figure"
    """"""
    DISABLE_FIGURE = "disable_figure"
    """"""
    RESTORE_FIGURE = "restore_figure"
    """"""
    CREATE_CLASS = "create_class"
    """"""
    UPDATE_CLASS = "update_class"
    """"""
    DISABLE_CLASS = "disable_class"
    """"""
    RESTORE_CLASS = "restore_class"
    """"""
    CREATE_BACKUP = "create_backup"
    """"""
    EXPORT_PROJECT = "export_project"
    """"""
    MODEL_TRAIN = "model_train"
    """"""
    MODEL_INFERENCE = "model_inference"
    """"""
    CREATE_PLUGIN = "create_plugin"
    """"""
    DISABLE_PLUGIN = "disable_plugin"
    """"""
    RESTORE_PLUGIN = "restore_plugin"
    """"""
    CREATE_NODE = "create_node"
    """"""
    DISABLE_NODE = "disable_node"
    """"""
    RESTORE_NODE = "restore_node"
    """"""
    CREATE_WORKSPACE = "create_workspace"
    """"""
    DISABLE_WORKSPACE = "disable_workspace"
    """"""
    RESTORE_WORKSPACE = "restore_workspace"
    """"""
    CREATE_MODEL = "create_model"
    """"""
    DISABLE_MODEL = "disable_model"
    """"""
    RESTORE_MODEL = "restore_model"
    """"""
    ADD_MEMBER = "add_member"
    """"""
    REMOVE_MEMBER = "remove_member"
    """"""
    LOGIN_TO_TEAM = "login_to_team"
    """"""
    ATTACH_TAG = "attach_tag"
    """"""
    UPDATE_TAG_VALUE = "update_tag_value"
    """"""
    DETACH_TAG = "detach_tag"
    """"""
    ANNOTATION_DURATION = "annotation_duration"
    """"""
    IMAGE_REVIEW_STATUS_UPDATED = "image_review_status_updated"
    """"""

    # case #1 - labeler pressed "finish image" button in labeling job
    # action: IMAGE_REVIEW_STATUS_UPDATED -> meta["reviewStatus"] == 'done'

    # case #2 - reviewer pressed "accept" or "reject" button
    # action: IMAGE_REVIEW_STATUS_UPDATED -> meta["reviewStatus"] == 'accepted' or 'rejected'

    # possible review statuses:
    # 'done' - i.e. labeler finished the image,
    # 'accepted' - reviewer
    # 'rejected' - reviewer

    # case #3 duration
    # action: ANNOTATION_DURATION -> meta["duration"] e.g. meta-> {"duration": 30} in seconds


class TeamInfo(NamedTuple):
    """ """

    id: int
    name: str
    description: str
    role: str
    created_at: str
    updated_at: str


class TeamApi(ModuleNoParent, UpdateableModule):
    """
    API for working with Team. :class:`TeamApi<TeamApi>` object is immutable.

    :param api: API connection to the server
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

        team_info = api.team.get_info_by_id(team_id) # api usage example
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple TeamInfo containing information about Team.

        :Example:

         .. code-block:: python

            TeamInfo(id=1,
                     name='Vehicle',
                     description='',
                     role='admin',
                     created_at='2020-03-31T14:49:08.931Z',
                     updated_at='2020-03-31T14:49:08.931Z')
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.DESCRIPTION,
            ApiField.ROLE,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **TeamInfo**.
        """
        return "TeamInfo"

    def __init__(self, api):
        ModuleNoParent.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(self, filters: List[Dict[str, str]] = None) -> List[TeamInfo]:
        """
        List of all Teams.

        :param filters: List of params to sort output Teams.
        :type filters: list, optional
        :return: List of all Teams with information. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[TeamInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            team_id = 8

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_list = api.team.get_list(team_id)
            print(team_list)
            # Output: [TeamInfo(id=1,
            #                   name='Vehicle',
            #                   description='',
            #                   role='admin',
            #                   created_at='2020-03-31T14:49:08.931Z',
            #                   updated_at='2020-03-31T14:49:08.931Z'),
            # TeamInfo(id=2,
            #          name='Road',
            #          description='',
            #          role='admin',
            #          created_at='2020-03-31T08:52:11.000Z',
            #          updated_at='2020-03-31T08:52:11.000Z'),
            # TeamInfo(id=3,
            #          name='Animal',
            #          description='',
            #          role='admin',
            #          created_at='2020-04-02T08:59:03.717Z',
            #          updated_at='2020-04-02T08:59:03.717Z')
            # ]

            # Filtered Team list
            team_list = api.team.get_list(team_id, filters=[{ 'field': 'name', 'operator': '=', 'value': 'Animal' }])
            print(team_list)
            # Output: [TeamInfo(id=3,
            #                  name='Animal',
            #                  description='',
            #                  role='admin',
            #                  created_at='2020-04-02T08:59:03.717Z',
            #                  updated_at='2020-04-02T08:59:03.717Z')
            # ]
        """
        return self.get_list_all_pages("teams.list", {ApiField.FILTER: filters or []})

    def get_info_by_id(self, id: int, raise_error: Optional[bool] = False) -> TeamInfo:
        """
        Get Team information by ID.

        :param id: Team ID in Supervisely.
        :type id: int
        :return: Information about Team. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`TeamInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_info = api.team.get_info_by_id(8)
            print(team_info)
            # Output: TeamInfo(id=8,
            #          name='Fruits',
            #          description='',
            #          role='admin',
            #          created_at='2020-04-15T10:50:41.926Z',
            #          updated_at='2020-04-15T10:50:41.926Z')

            # You can also get Team info by name
            team_info = api.team.get_info_by_name("Fruits")
            print(team_info)
            # Output: TeamInfo(id=8,
            #          name='Fruits',
            #          description='',
            #          role='admin',
            #          created_at='2020-04-15T10:50:41.926Z',
            #          updated_at='2020-04-15T10:50:41.926Z')
        """

        info = self._get_info_by_id(id, "teams.info")
        if info is None and raise_error is True:
            raise KeyError(f"Team with id={id} not found in your account")
        return info

    def create(
        self,
        name: str,
        description: Optional[str] = "",
        change_name_if_conflict: Optional[bool] = False,
    ) -> TeamInfo:
        """
        Creates Team with given name.

        :param name: Team name.
        :type name: str
        :param description: Team description.
        :type description: str
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :return: Information about Team. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`TeamInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_team = api.team.create("Flowers")
            print(new_team)
            # Output: TeamInfo(id=228,
            #                  name='Flowers',
            #                  description='',
            #                  role='admin',
            #                  created_at='2021-03-11T11:18:46.576Z',
            #                  updated_at='2021-03-11T11:18:46.576Z')
        """
        effective_name = self._get_effective_new_name(
            name=name, change_name_if_conflict=change_name_if_conflict
        )
        response = self._api.post(
            "teams.add",
            {ApiField.NAME: effective_name, ApiField.DESCRIPTION: description},
        )
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        """ """
        return "teams.editInfo"

    def get_activity(
        self,
        team_id: int,
        filter_user_id: Optional[int] = None,
        filter_project_id: Optional[int] = None,
        filter_job_id: Optional[int] = None,
        filter_actions: Optional[List] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get Team activity by ID.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param filter_user_id: User ID by which the activity will be filtered.
        :type filter_user_id: int, optional
        :param filter_project_id: Project ID by which the activity will be filtered.
        :type filter_project_id: int, optional
        :param filter_job_id: Job ID by which the activity will be filtered.
        :type filter_job_id: int, optional
        :param filter_actions: List of ActivityAction by which the activity will be filtered.
        :type filter_actions: list, optional
        :param progress_cb: Function to check progress.
        :type progress_cb: tqdm or callable, optional
        :param start_date: Start date to get Team activity.
        :type start_date: str, optional
        :param end_date: End date to get Team activity.
        :type end_date: str, optional
        :return: Team activity
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.api.team_api import ActivityAction as aa

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            labeling_actions = [
                aa.ATTACH_TAG,
                aa.UPDATE_TAG_VALUE,
                aa.DETACH_TAG,
            ]

            team_activity = api.team.get_activity(8, filter_actions=labeling_actions)
            print(team_activity)
            # Output: [
            #     {
            #         "userId":7,
            #         "action":"detach_tag",
            #         "date":"2021-01-15T15:11:55.985Z",
            #         "user":"cxnt",
            #         "projectId":1817,
            #         "project":"App_Test_Poly",
            #         "datasetId":2370,
            #         "dataset":"train",
            #         "imageId":726985,
            #         "image":"IMG_8144.jpeg",
            #         "classId":"None",
            #         "class":"None",
            #         "figureId":"None",
            #         "job":"None",
            #         "jobId":"None",
            #         "tag":"hhhlk",
            #         "tagId":4720,
            #         "meta":{}
            #     },
            #     {
            #         "userId":7,
            #         "action":"attach_tag",
            #         "date":"2021-01-15T14:24:58.480Z",
            #         "user":"cxnt",
            #         "projectId":1817,
            #         "project":"App_Test_Poly",
            #         "datasetId":2370,
            #         "dataset":"train",
            #         "imageId":726985,
            #         "image":"IMG_8144.jpeg",
            #         "classId":"None",
            #         "class":"None",
            #         "figureId":"None",
            #         "job":"None",
            #         "jobId":"None",
            #         "tag":"hhhlk",
            #         "tagId":4720,
            #         "meta":{}
            #     }
            # ]
        """
        from datetime import datetime, timedelta

        filters = []
        if filter_user_id is not None:
            filters.append({"field": ApiField.USER_ID, "operator": "=", "value": filter_user_id})
        if filter_project_id is not None:
            filters.append(
                {
                    "field": ApiField.PROJECT_ID,
                    "operator": "=",
                    "value": filter_project_id,
                }
            )
        if filter_job_id is not None:
            filters.append({"field": ApiField.JOB_ID, "operator": "=", "value": filter_job_id})
        if filter_actions is not None:
            if type(filter_actions) is not list:
                raise TypeError(
                    "type(filter_actions) is {!r}. But has to be of type {!r}".format(
                        type(filter_actions), list
                    )
                )
            filters.append({"field": ApiField.TYPE, "operator": "in", "value": filter_actions})

        def _add_dt_filter(filters, dt, op):
            dt_iso = None
            if dt is None:
                return
            if type(dt) is str:
                dt_iso = dt
            elif type(dt) is datetime:
                dt_iso = dt.isoformat()
            else:
                raise TypeError(
                    "DT type must be string in ISO8601 format or datetime, not {}".format(type(dt))
                )
            filters.append({"field": ApiField.DATE, "operator": op, "value": dt_iso})

        _add_dt_filter(filters, start_date, ">=")
        _add_dt_filter(filters, end_date, "<=")

        method = "teams.activity"
        data = {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters}
        first_response = self._api.post(method, data)
        first_response = first_response.json()

        total = first_response["total"]
        per_page = first_response["perPage"]
        pages_count = first_response["pagesCount"]
        results = first_response["entities"]

        def set_tqdm(progress_cb, results, total):
            progress_cb.total = total
            progress_cb.update(len(results) - progress_cb.n)
            progress_cb.refresh()

        if progress_cb is not None:
            if isinstance(progress_cb, tqdm):
                set_tqdm(progress_cb, results, total)
            else:
                progress_cb(len(results), total)
        if pages_count == 1 and len(first_response["entities"]) == total:
            pass
        else:
            for page_idx in range(2, pages_count + 1):
                temp_resp = self._api.post(method, {**data, "page": page_idx, "per_page": per_page})
                temp_items = temp_resp.json()["entities"]
                results.extend(temp_items)
                if progress_cb is not None:
                    if isinstance(progress_cb, tqdm):
                        set_tqdm(progress_cb, results, total)
                    else:
                        progress_cb(len(results), total)
            if len(results) != total:
                logger.warn(
                    f"Method '{method}': new events were created during pagination, "
                    f"downloaded={len(results)}, total={total}"
                )

        return results

    def _convert_json_info(self, info: dict, skip_missing=True) -> TeamInfo:
        """ """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return TeamInfo(**res._asdict())
