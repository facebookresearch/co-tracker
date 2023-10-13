from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Carousel(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(self, name: str = "", label: str = "", is_link: bool = True) -> Carousel.Item:
            self.name = name
            self.label = label
            self.is_link = is_link

        def to_json(self):
            return {
                "name": self.name,
                "label": self.label,
                "is_link": self.is_link,
            }

    def __init__(
        self,
        items: List[Carousel.Item],
        height: int = 350,
        initial_index: int = 0,
        trigger: Literal["hover", "click"] = "click",
        autoplay: bool = False,
        interval: int = 3000,
        indicator_position: Union[Literal["outside", "none"], None] = "none",
        arrow: Literal["always", "hover", "never"] = "hover",
        type: Union[Literal["card"], None] = None,
        widget_id: str = None,
    ):
        self._height = f"{height}px"
        self._items = items
        self._initial_index = initial_index
        self._trigger = trigger
        self._autoplay = autoplay
        self._interval = interval
        self._indicator_position = indicator_position if indicator_position is not None else "none"
        self._arrow = arrow
        self._type = type

        self._changes_handled = False
        self._clicked_value = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_items(self):
        return [item.to_json() for item in self._items]

    def get_json_data(self):
        return {
            "height": self._height,
            "items": self._set_items(),
            "initialIndex": self._initial_index,
            "trigger": self._trigger,
            "autoplay": self._autoplay,
            "interval": self._interval,
            "indicatorPosition": self._indicator_position,
            "arrow": self._arrow,
            "type": self._type,
        }

    def get_json_state(self):
        return {"clickedValue": self._clicked_value}

    def get_active_item(self):
        return StateJson()[self.widget_id]["clickedValue"]

    def get_items(self):
        return DataJson()[self.widget_id]["items"]

    def set_items(self, value: List[Carousel.Item]):
        if not all(isinstance(item, Carousel.Item) for item in value):
            raise ValueError("Items must be of type Carousel.Item")
        self._items = value
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def add_items(self, value: List[Carousel.Item]):
        self._items.extend(value)
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def set_height(self, value: int):
        if type(value) is not int:
            raise ValueError("Height value must be an integer")
        self._height = f"{value}px"
        DataJson()[self.widget_id]["height"] = self._height
        DataJson().send_changes()

    def get_initial_index(self):
        return DataJson()[self.widget_id]["initialIndex"]

    def set_initial_index(self, value: int):
        if value >= len(self._items):
            raise ValueError("Index of the value being set exceeds the size of the carousel items.")
        self._initial_index = value
        DataJson()[self.widget_id]["initialIndex"] = self._initial_index
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Carousel.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            curr_idx = self.get_active_item()
            curr_name = self._items[curr_idx].name
            func(curr_name)

        return _click
