# coding: utf-8
"""list available classes in supervisely project"""

# docs
from typing import Optional, List, Dict

from supervisely.api.module_api import ModuleApi
from supervisely.api.module_api import ApiField


class ProjectClassApi(ModuleApi):
    """
    API for working with classes in Project. :class:`ProjectClassApi<ProjectClassApi>` object is immutable.

    :param api: API connection to the server
    :type api: Api
    """
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                #ApiField.PROJECT_ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.SHAPE,
                ApiField.COLOR,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT
                ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **ProjectClassInfo**.
        """
        return 'ProjectClassInfo'

    def get_list(self, project_id: int, filters: Optional[List[Dict[str, str]]] = None) -> list:
        """
        List of Classes in the given Project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param filters:
        :type filters: list
        :return: List of classes.
        :rtype: :class:`list`
        """
        return self.get_list_all_pages('advanced.object_classes.list',  {ApiField.PROJECT_ID: project_id, "filter": filters or []})
