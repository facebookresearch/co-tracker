# coding: utf-8
"""list/create supervisely workspaces"""

from __future__ import annotations
from typing import NamedTuple, List, Dict, Optional

from supervisely.api.module_api import ApiField, ModuleApi, UpdateableModule


class WorkspaceInfo(NamedTuple):
    """ """

    id: int
    name: str
    description: str
    team_id: int
    created_at: str
    updated_at: str


class WorkspaceApi(ModuleApi, UpdateableModule):
    """
    API for working with Workspace. :class:`WorkspaceApi<WorkspaceApi>` object is immutable.

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

        workspace_info = api.workspace.get_info_by_id(workspace_id) # api usage example
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple WorkspaceInfo containing information about Workspace.

        :Example:

         .. code-block:: python

            WorkspaceInfo(id=15,
                          name='Cars',
                          description='Workspace contains Project with annotated Cars',
                          team_id=8,
                          created_at='2020-04-15T10:50:41.926Z',
                          updated_at='2020-04-15T10:50:41.926Z')
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.DESCRIPTION,
            ApiField.TEAM_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **WorkspaceInfo**.
        """
        return "WorkspaceInfo"

    def __init__(self, api):
        ModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(
        self, team_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[WorkspaceInfo]:
        """
        List of Workspaces in the given Team.

        :param team_id: Team ID in which the Workspaces are located.
        :type team_id: int
        :param filters: List of params to sort output Workspaces.
        :type filters: List[dict], optional
        :return: List of all Workspaces with information for the given Team. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[WorkspaceInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            workspace_infos = api.workspace.get_list(8)
            print(workspace_infos)
            # Output: [
            # WorkspaceInfo(id=15,
            #               name='Cars',
            #               description='',
            #               team_id=8,
            #               created_at='2020-04-15T10:50:41.926Z',
            #               updated_at='2020-04-15T10:50:41.926Z'),
            # WorkspaceInfo(id=18,
            #               name='Heart',
            #               description='',
            #               team_id=8,
            #               created_at='2020-05-20T15:01:54.172Z',
            #               updated_at='2020-05-20T15:01:54.172Z'),
            # WorkspaceInfo(id=20,
            #               name='PCD',
            #               description='',
            #               team_id=8,
            #               created_at='2020-06-24T11:51:11.336Z',
            #               updated_at='2020-06-24T11:51:11.336Z')
            # ]

            # Filtered Workspace list
            workspace_infos = api.workspace.get_list(8, filters=[{ 'field': 'name', 'operator': '=', 'value': 'Heart'}])
            print(workspace_infos)
            # Output: [WorkspaceInfo(id=18,
            #                       name='Heart',
            #                       description='',
            #                       team_id=8,
            #                       created_at='2020-05-20T15:01:54.172Z',
            #                       updated_at='2020-05-20T15:01:54.172Z')
            # ]
        """
        return self.get_list_all_pages(
            "workspaces.list",
            {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters or []},
        )

    def get_info_by_id(self, id: int, raise_error: Optional[bool] = False) -> WorkspaceInfo:
        """
        Get Workspace information by ID.

        :param id: Workspace ID in Supervisely.
        :type id: int
        :return: Information about Workspace. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`WorkspaceInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            workspace_info = api.workspace.get_info_by_id(58)
            print(workspace_info)
            # Output: WorkspaceInfo(id=58,
            #                       name='Test',
            #                       description='',
            #                       team_id=8,
            #                       created_at='2020-11-09T18:21:08.202Z',
            #                       updated_at='2020-11-09T18:21:08.202Z')
        """
        info = self._get_info_by_id(id, "workspaces.info")
        if info is None and raise_error is True:
            raise KeyError(f"Workspace with id={id} not found in your account")
        return info

    def create(
        self,
        team_id: int,
        name: str,
        description: Optional[str] = "",
        change_name_if_conflict: Optional[bool] = False,
    ) -> WorkspaceInfo:
        """
        Create Workspace with given name in the given Team.

        :param team_id: Team ID in Supervisely where Workspace will be created.
        :type team_id: int
        :param name: Workspace Name.
        :type name: str
        :param description: Workspace description.
        :type description: str, optional
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :return: Information about Workspace. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`WorkspaceInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_workspace = api.workspace.create(8, "Vehicle Detection")
            print(new_workspace)
            # Output: WorkspaceInfo(id=274,
            #                       name='Vehicle Detection"',
            #                       description='',
            #                       team_id=8,
            #                       created_at='2021-03-11T12:24:21.773Z',
            #                       updated_at='2021-03-11T12:24:21.773Z')
        """
        effective_name = self._get_effective_new_name(
            parent_id=team_id,
            name=name,
            change_name_if_conflict=change_name_if_conflict,
        )
        response = self._api.post(
            "workspaces.add",
            {
                ApiField.TEAM_ID: team_id,
                ApiField.NAME: effective_name,
                ApiField.DESCRIPTION: description,
            },
        )
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        """ """
        return "workspaces.editInfo"

    def _convert_json_info(self, info: dict, skip_missing=True) -> WorkspaceInfo:
        """ """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return WorkspaceInfo(**res._asdict())
