from __future__ import annotations
from supervisely.app import StateJson
from supervisely.app.widgets import Widget, Text
from typing import List, Dict, Union


class Checkbox(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        content: Union[Widget, str],
        checked: bool = False,
        widget_id: str = None,
    ):
        self._content = content
        self._checked = checked
        if type(self._content) is str:
            self._content = [Text(self._content)][0]
        self._changes_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {"checked": self._checked}

    def is_checked(self):
        return StateJson()[self.widget_id]["checked"]

    def _set(self, checked: bool):
        self._checked = checked
        StateJson()[self.widget_id]["checked"] = self._checked
        StateJson().send_changes()

    def check(self):
        self._set(True)

    def uncheck(self):
        self._set(False)

    def value_changed(self, func):
        route_path = self.get_route_path(Checkbox.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.is_checked()
            func(res)

        return _click