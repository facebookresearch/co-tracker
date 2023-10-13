from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, ConditionalWidget, ConditionalItem
from typing import List, Dict, Union


class Switch(ConditionalWidget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
            self,
            switched: bool = False,
            width: int = 58,  # default value 46 for empty text
            on_text: str = "ON",
            off_text: str = "OFF",
            on_color: str = None,  # default: "#20a0ff"
            off_color: str = None,  # default: "#bfcbd9"
            on_content: Widget = None,
            off_content: Widget = None,
            widget_id: str = None,
    ):
        self._switched = switched
        self._width = width
        self._on_text = on_text
        self._off_text = off_text
        self._on_color = on_color
        self._off_color = off_color
        self._changes_handled = False
        items = [
            ConditionalItem(value=True, content=on_content),
            ConditionalItem(value=False, content=off_content)
        ]
        super().__init__(items=items, widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "width": self._width,
            "onText": self._on_text,
            "offText": self._off_text,
            "onColor": self._on_color,
            "offColor": self._off_color
        }

    def get_json_state(self) -> Dict:
        return {"value": self._switched}

    def is_switched(self):
        return StateJson()[self.widget_id]["value"]

    def on(self):
        StateJson()[self.widget_id]["value"] = True
        StateJson().send_changes()

    def off(self):
        StateJson()[self.widget_id]["value"] = False
        StateJson().send_changes()

    def get_width(self):
        return DataJson()[self.widget_id]["width"]

    def get_on_text(self):
        return DataJson()[self.widget_id]["onText"]

    def set_on_text(self, value: str):
        DataJson()[self.widget_id]["onText"] = value
        DataJson().send_changes()

    def get_off_text(self):
        return DataJson()[self.widget_id]["offText"]

    def set_off_text(self, value: str):
        DataJson()[self.widget_id]["offText"] = value
        DataJson().send_changes()

    def get_on_color(self):
        return DataJson()[self.widget_id]["onColor"]

    def set_on_color(self, value: str):
        DataJson()[self.widget_id]["onColor"] = value
        DataJson().send_changes()

    def get_off_color(self):
        return DataJson()[self.widget_id]["offColor"]

    def set_off_color(self, value: str):
        DataJson()[self.widget_id]["offColor"] = value
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Switch.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.is_switched()
            func(res)

        return _click
