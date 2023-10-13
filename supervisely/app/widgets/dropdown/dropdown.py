from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Dropdown(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(
            self,
            text: str = "",
            disabled: bool = False,
            divided: bool = False,
            command: Union[str, int] = None,
        ) -> Dropdown.Item:
            self.text = text
            self.disabled = disabled
            self.divided = divided
            self.command = command

        def to_json(self):
            return {
                "text": self.text,
                "disabled": self.disabled,
                "divided": self.divided,
                "command": self.command,
            }

    def __init__(
        self,
        items: List[Dropdown.Item] = None,
        header: str = "Dropdown List",
        trigger: Literal["hover", "click"] = "click",
        menu_align: Literal["start", "end"] = "end",
        hide_on_click: bool = True,
        widget_id: str = None,
    ):
        self._items = items
        self._header = header
        self._trigger = trigger
        self._menu_align = menu_align
        self._hide_on_click = hide_on_click
        self._changes_handled = False
        self._clicked_value = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_items(self):
        return [item.to_json() for item in self._items]

    def get_json_data(self):
        return {
            "trigger": self._trigger,
            "items": self._set_items(),
            "menuAlign": self._menu_align,
            "hideOnClick": self._hide_on_click,
            "header": self._header,
        }

    def get_json_state(self):
        return {"clickedValue": self._clicked_value}

    def get_value(self):
        return StateJson()[self.widget_id]["clickedValue"]

    def set_value(self, value: str):
        self._clicked_value = value
        StateJson()[self.widget_id]["clickedValue"] = self._clicked_value
        StateJson().send_changes()

    def get_items(self):
        return DataJson()[self.widget_id]["items"]

    def set_items(self, value: List[Dropdown.Item]):
        if not all(isinstance(item, Dropdown.Item) for item in value):
            raise TypeError("Items must be a list of Dropdown.Item")
        self._items = value
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def add_items(self, value: List[Dropdown.Item]):
        self._items.extend(value)
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def get_header_text(self):
        return DataJson()[self.widget_id]["header"]

    def set_header_text(self, value: str):
        if type(value) is not str:
            raise TypeError("Header value must be a string")
        self._header = value
        DataJson()[self.widget_id]["header"] = self._header
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Dropdown.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            func(res)

        return _click
