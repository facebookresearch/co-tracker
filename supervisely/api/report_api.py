# coding: utf-8

import os
import json
import urllib.parse
import uuid
from typing import List, NamedTuple, Dict, Optional

from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.collection.str_enum import StrEnum


class NotificationType(StrEnum):
    """
    """

    INFO = 'info'
    """"""
    NOTE = "note"
    """"""
    WARNING = "warning"
    """"""
    ERROR = "error"
    """"""


# @TODO: стандартизовать title/description/name и так жалее у всех одинакого
class ReportApi(ModuleApiBase):
    """
    API for working with Reports. :class:`ReportApi<ReportApi>` object is immutable.

    :param api: API connection to the server
    :type api: Api
    :Usage example:

     .. code-block:: python

        report = api.report
    """

    def __init__(self, api):
        ModuleApiBase.__init__(self, api)

    # https://developer.mozilla.org/en-US/docs/Web/CSS/grid-template
    # grid-template: "a a a" 40px "b c c" 40px "b c c" 40px / 1fr 1fr 1fr;
    # area -a or b or c
    def create(self, team_id: int, name: str, widgets, layout: Optional[str] = ""):
        """
        Creates report in the given Team.

        :param team_id: Team ID in Supervisely, where report will be created
        :type team_id: int
        :param name: Report name.
        :type name: str
        :param widgets:
        :type widgets:
        :param layout:
        :type layout:
        :return:
        :rtype:
        :Usage example:

         .. code-block: python
        """
        data = {
            ApiField.TEAM_ID: team_id,
            ApiField.NAME: name,
            ApiField.WIDGETS: widgets,
            ApiField.LAYOUT: layout
        }
        response = self._api.post('reports.create', data)
        return response.json()[ApiField.ID]

    # def create_table(self, df, name, subtitle, per_page=20, pageSizes=[10, 20, 50, 100, 500], fix_columns=None):
    #     res = {
    #         "name": name,
    #         "subtitle": subtitle,
    #         "type": str(WidgetType.TABLE),
    #         "content": json.loads(df.to_json(orient='split')),
    #         "options": {
    #             "perPage": per_page,
    #             "pageSizes": pageSizes,
    #         }
    #     }
    #     if fix_columns is not None:
    #         res["options"]["fixColumns"] = fix_columns
    #     return res
    #
    # def create_notification(self, name, content, notification_type=NotificationType.INFO):
    #     return {
    #         "type": str(WidgetType.NOTIFICATION),
    #         "title": name,
    #         "content": content,
    #         "options": {
    #             "type": str(notification_type)
    #         }
    #     }
    #
    # def create_plotly(self, data_json, name, subtitle):
    #     data = data_json
    #     if type(data) is str:
    #         data = json.loads(data_json)
    #     elif type(data) is not dict:
    #         raise RuntimeError("type(data_json) is not dict")
    #     return {
    #         "name": name,
    #         "subtitle": subtitle,
    #         "type": str(WidgetType.PLOTLY),
    #         "content": data
    #     }
    #
    #
    # def create_linechart(self, name, description, id=None):
    #     res = {
    #         "type": str(WidgetType.LINECHART),
    #         "name": "linechart block title",
    #         "subtitle": "linechart block description",
    #         "content": [],
    #         "options": {}
    #     }
    #     res["id"] = uuid.uuid4().hex if id is None else id
    #     return res

    def url(self, id: int) -> str:
        """
        Get Report URL by ID.

        :param id: Report ID.
        :type id: int
        :returns: Report URL
        :rtype: :class:`str`
        """
        return urllib.parse.urljoin(self._api.server_address, 'reports/{}'.format(id))

    def get_widget(self, report_id: int, widget_id: int):
        """
        Get Widget by ID.

        :param report_id: Report ID.
        :type report_id: int
        :param widget_id: Widget ID.
        :type widget_id: int
        :returns: Report Widget
        :rtype:
        """
        response = self._api.post('reports.widgets.get', {"reportId": report_id, "widgetId": widget_id})
        return response.json()

    def _change_widget(self, method, report_id, widget_id, widget_type=None, name=None, description=None, area=None,
                       content=None, options=None):
        """
        """
        data = dict()
        data[ApiField.ID] = widget_id
        if name is not None:
            data[ApiField.NAME] = name
        if widget_type is not None:
            data[ApiField.TYPE] = widget_type
        if description is not None:
            data[ApiField.SUBTITLE] = description
        if area is not None:
            data[ApiField.AREA] = area
        if content is not None:
            data[ApiField.CONTENT] = content
        if options is not None:
            data[ApiField.OPTIONS] = options
        response = self._api.post(method, {ApiField.REPORT_ID: report_id, ApiField.WIDGET: data})
        return response.json()

    def update_widget(self, report_id: int, widget_id: int, name: Optional[str] = None,
                      description: Optional[str] = None,
                      area=None, content=None, options=None):
        """
        Method description

        :param report_id: Report ID.
        :type report_id: int
        :param widget_id: Widget ID.
        :type widget_id: int
        :param name:
        :type name: str
        :param description:
        :type description: str
        :param area:
        :type area:
        :param content:
        :type content:
        :param options:
        :type options:

        :returns: Report Widget
        :rtype:
        """
        return self._change_widget('reports.widgets.update', report_id, widget_id, name, description, area, content,
                                   options)

    def rewrite_widget(self, report_id: int, widget_id: int, widget_type, name: str = None, description: str = None,
                       area=None, content=None, options=None):
        """
        Method description

        :param report_id: Report ID.
        :type report_id: int
        :param widget_id: Widget ID.
        :type widget_id: int
        :param widget_type: Widget type.
        :type widget_type:
        :param name:
        :type name: str
        :param description:
        :type description: str
        :param area:
        :type area:
        :param content:
        :type content:
        :param options:
        :type options:

        :returns: Report Widget
        :rtype:
        """
        return self._change_widget('reports.widgets.rewrite', report_id, widget_id, widget_type, name, description,
                                   area, content, options)
