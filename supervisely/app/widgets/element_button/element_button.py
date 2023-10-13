import copy
import functools
from enum import Enum

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import fastapi
from varname import varname

from supervisely.app import DataJson
from supervisely.app.fastapi import run_sync
from supervisely.app.widgets import Widget


class ElementButton(Widget):
    class Routes:
        BUTTON_CLICKED = "button_clicked_cb"

    def __init__(
        self,
        text: str = "Element Button",
        button_type: Literal[
            "primary", "info", "warning", "danger", "success"
        ] = "primary",
        button_size: Literal["mini", "small", "large"] = None,
        plain: bool = False,
        widget_id: str = None,
    ):
        self._widget_routes = {}

        self._text = text
        self._button_type = button_type
        self._plain = plain
        self._button_size = button_size

        self._loading = False
        self._disabled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "text": self._text,
            "button_type": self._button_type,
            "plain": self._plain,
            "button_size": self._button_size,
            "loading": self._loading,
            "disabled": self._disabled,
        }

    def get_json_state(self):
        return None

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        DataJson()[self.widget_id]["text"] = self._text

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value):
        self._disabled = value
        DataJson()[self.widget_id]["disabled"] = self._disabled

    def click(self):
        route = ElementButton.Routes.BUTTON_CLICKED
        return self.add_route(self._sly_app.get_server(), route)
