from typing import Optional, Union

from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget


class Badge(Widget):
    def __init__(
        self,
        value: Union[int, str, float] = None,
        widget: Optional[Widget] = None,
        max: Union[int, float] = None,
        is_dot: bool = False,
        hidden: bool = False,
        widget_id: str = None,
    ):
        self._value = value
        self._widget = widget
        self._max = max if type(max) in [int, float] else None
        self._hidden = hidden
        self._is_dot = is_dot

        if self._value is None and self._widget is not None:
            self._is_dot = True

        if self._is_dot is True and self._value is None:
            self._value = 0

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        res = {}
        res["max"] = self._max
        res["isDot"] = self._is_dot
        res["hidden"] = self._hidden
        return res

    def get_json_state(self):
        return {"value": self._value}

    def set_value(self, value: Union[str, int, float]) -> None:
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_value(self) -> Union[str, int, float]:
        if "value" not in StateJson()[self.widget_id].keys():
            return None
        value = StateJson()[self.widget_id]["value"]
        return value

    def clear(self):
        self._value = None
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def hide_badge(self):
        self._hidden = True
        DataJson()[self.widget_id]["hidden"] = self._hidden
        DataJson().send_changes()

    def show_badge(self):
        self._hidden = False
        DataJson()[self.widget_id]["hidden"] = self._hidden
        DataJson().send_changes()

    def toggle_visibility(self):
        self._hidden = not self._hidden
        DataJson()[self.widget_id]["hidden"] = self._hidden
        DataJson().send_changes()
