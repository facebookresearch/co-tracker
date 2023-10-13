# coding: utf-8
"""api for working with agent"""

from __future__ import annotations

from typing import NamedTuple, Optional, Dict, List
from enum import Enum
from supervisely.api.module_api import ApiField, ModuleApi, ModuleWithStatus


class AgentNotFound(Exception):
    """class AgentNotFound"""

    pass


class AgentNotRunning(Exception):
    """class AgentNotRunning"""

    pass


class AgentApi(ModuleApi, ModuleWithStatus):
    """
    API for working with agent. :class:`AgentApi<AgentApi>` object is immutable.

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

        team_id = 8
        agents = api.agent.get_list(team_id)
    """

    class Status(Enum):
        """Agent status."""

        WAITING = "waiting"
        """"""
        RUNNING = "running"
        """"""

    @staticmethod
    def info_sequence():
        """
        NamedTuple AgentInfo information about Agent.

        :Example:

         .. code-block:: python

            AgentInfo("some info")
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.TOKEN,
            ApiField.STATUS,
            ApiField.USER_ID,
            ApiField.TEAM_ID,
            ApiField.CAPABILITIES,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **AgentInfo**.
        """
        return "AgentInfo"

    def __init__(self, api):
        ModuleApi.__init__(self, api)
        ModuleWithStatus.__init__(self)

    def get_list(
        self, team_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[NamedTuple]:
        """
        List of all agents in the given Team.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param filters: List of params to sort output Agents.
        :type filters: List[dict], optional
        :return: List of Agents with information. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 16087
            agents = api.agent.get_list(team_id)

            filter_agents = api.agent.get_list(team_id, filters=[{ 'field': 'name', 'operator': '=', 'value': 'Gorgeous Chicken' }])
        """
        return self.get_list_all_pages("agents.list", {"teamId": team_id, "filter": filters or []})

    def get_info_by_id(self, id: int) -> NamedTuple:
        """
        Get Agent information by ID.

        :param id: Agent ID in Supervisely.
        :type id: int
        :return: Information about Agent. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            agent = api.agent.get_info_by_id(7)
        """
        return self._get_info_by_id(id, "agents.info")

    def get_status(self, id: int) -> AgentApi.Status:
        """
        Status object containing status of Agent: waiting or running.

        :param id: Agent ID in Supervisely.
        :type id: int
        :return: Agent Status
        :rtype: :class:`Status<supervisely.api.agent_api.AgentApi.Status>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            agent = api.agent.get_status(7)
        """
        status_str = self.get_info_by_id(id).status
        return self.Status(status_str)

    def raise_for_status(self, status):
        """raise_for_status"""
        pass
