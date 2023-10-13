from __future__ import annotations
from supervisely.app.jinja2 import create_env
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget, ConditionalWidget, ConditionalItem
from typing import List, Dict, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class RadioGroup(ConditionalWidget):

    class Routes:
        VALUE_CHANGED = "value_changed"
    
    class Item(ConditionalItem):
        pass

    def __init__(
        self,
        items: List[RadioGroup.Item],
        size: Literal["large", "small", "mini"] = None,
        direction: Literal["vertical", "horizontal"] = "horizontal",
        gap: int = 10,
        widget_id: str = None,
    ) -> RadioGroup:

        self._changes_handled = False
        self._size = size
        self._gap = gap
        self._value = None

        self._flex_direction = "row"
        if direction == "vertical":
            self._flex_direction = "column"

        super().__init__(items=items, widget_id=widget_id, file_path=__file__)

    def _get_first_value(self) -> RadioGroup.Item:
        if self._items is not None and len(self._items) > 0:
            return self._items[0]
        return None

    def get_json_data(self) -> Dict:
        res = {
            "items": None,
        }
        if self._items is not None:
            res["items"] = [item.to_json() for item in self._items]
        if self._size is not None:
            res["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        value = self._value
        if value is None:
            first_item = self._get_first_value()
            if first_item is not None:
                value = first_item.value
        return {"value": value}

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def value_changed(self, func):
        route_path = self.get_route_path(RadioGroup.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._value = res
            func(res)

        return _click

    def set(self, items: List[RadioGroup.Item]):
        self._items = items
        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()

    def set_value(self, value):
        self._value = value
        self.update_state()
        StateJson().send_changes()
