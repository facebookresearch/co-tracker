# coding: utf-8
"""get list of plugins for supervisely team"""

from __future__ import annotations
from typing import NamedTuple, List, Dict, Optional

from supervisely.api.module_api import ApiField, ModuleApi


class PluginApi(ModuleApi):
    """
    API for working with plugins. :class:`PluginApi<PluginApi>` object is immutable.

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
        plugin_info = api.plugin.get_list(team_id)
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple PluginInfo information about Plugin.

        :Example:

         .. code-block:: python

            PluginInfo(id=3,
                       name='DTL',
                       description='Allows to combine datasets, to make class mapping, filter objects and images, apply auto augmentations and so on ...',
                       type='dtl',
                       default_version='latest',
                       docker_image='docker.deepsystems.io/supervisely/five/dtl',
                       readme='# Data Transformation Language (DTL)...',
                       configs=[],
                       versions=['lately', 'docs', ...],
                       created_at='2020-03-30T09:17:36.000Z',
                       updated_at='2020-04-23T06:26:29.000Z')
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.DESCRIPTION,
            ApiField.TYPE,
            ApiField.DEFAULT_VERSION,
            ApiField.DOCKER_IMAGE,
            ApiField.README,
            ApiField.CONFIGS,
            ApiField.VERSIONS,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **PluginInfo**.
        """
        return "PluginInfo"

    def get_list(
        self, team_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[NamedTuple]:
        """
        Get list of plugins in the Team.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param filters: List of params to sort output Plugins.
        :type filters: List[dict], optional
        :return: List of Plugins with information. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 8
            plugin_info = api.plugin.get_list(team_id)

            plugin_list_filter = api.plugin.get_list(team_id, filters=[{'field': 'name', 'operator': '=', 'value': 'Images'}])
        """
        return self.get_list_all_pages(
            "plugins.list", {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters or []}
        )

    def get_info_by_id(self, team_id: int, plugin_id: int) -> NamedTuple:
        """
        Get Plugin information by ID.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param plugin_id: Plugin ID in Supervisely.
        :type plugin_id: int
        :return: Information about Plugin. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            plugin_info = api.plugin.get_info_by_id(8, 3)
            print(plugin_info)
            # Output: PluginInfo(id=3,
            #                    name='DTL',
            #                    description='Allows to combine datasets, to make class mapping, filter objects and images, apply auto augmentations and so on ...',
            #                    type='dtl',
            #                    default_version='latest',
            #                    docker_image='docker.deepsystems.io/supervisely/five/dtl',
            #                    readme='# Data Transformation Language (DTL)...',
            #                    configs=[],
            #                    versions=['lately', 'docs', ...],
            #                    created_at='2020-03-30T09:17:36.000Z',
            #                    updated_at='2020-04-23T06:26:29.000Z')
        """
        filters = [{"field": ApiField.ID, "operator": "=", "value": plugin_id}]
        return self._get_info_by_filters(team_id, filters)
