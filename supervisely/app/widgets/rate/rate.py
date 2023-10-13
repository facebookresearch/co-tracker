from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Union


class Rate(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: Union[int, float] = None,
        max: int = 5,
        disabled: bool = False,
        allow_half: bool = False,
        texts: List[str] = [],
        show_text: bool = False,
        text_color: str = "#1F2D3D",
        text_template: str = "",
        colors: List = ["#F7BA2A", "#F7BA2A", "#F7BA2A"],
        void_color: str = "#C6D1DE",
        disabled_void_color: str = "#EFF2F7",
        widget_id: str = None,
    ):
        self._value = value
        self._max = max
        self._disabled = disabled
        self._allow_half = allow_half

        # text properties
        self._texts = texts
        self._show_text = show_text
        self._text_color = text_color
        self._text_template = text_template  # available if self._disabled is True

        # color properties
        self._colors = colors
        self._void_color = void_color
        self._disabled_void_color = disabled_void_color


        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "max": self._max,
            "disabled": self._disabled,
            "allowHalf": self._allow_half,
            "texts": self._texts,
            "showText": self._show_text,
            "textColor": self._text_color,
            "textTemplate": self._text_template,
            "colors": self._colors,
            "voidColor": self._void_color,
            "disabledVoidColor": self._disabled_void_color,
        }

    def get_json_state(self):
        return {"value": self._value}

    @property
    def is_disabled(self):
        self._disabled = DataJson()[self.widget_id]["disabled"]
        return self._disabled

    def get_value(self):
        self._value = StateJson()[self.widget_id]["value"]
        return self._value

    def set_value(self, value: Union[int, float]):
        if not isinstance(value, (int, float)):
            raise TypeError("Value type has to be one of int or float.")
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_max_value(self):
        self._max = DataJson()[self.widget_id]["max"]
        return self._max

    def set_max_value(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Max value type has to be a integer.")
        self._max = value
        DataJson()[self.widget_id]["max"] = self._max
        DataJson().send_changes()

    def get_colors(self):
        self._colors = DataJson()[self.widget_id]["colors"]
        return self._colors

    def set_colors(self, value: List[str]):
        if not isinstance(value, list):
            raise TypeError("Argument value type has to be a list.")
        self._colors = value
        DataJson()[self.widget_id]["colors"] = self._colors
        DataJson().send_changes()

    def allow_half_precision(self):
        self._allow_half = True
        DataJson()[self.widget_id]["allow_half"] = self._allow_half
        DataJson().send_changes()

    def disallow_half_precision(self):
        self._allow_half = False
        DataJson()[self.widget_id]["allow_half"] = self._allow_half
        DataJson().send_changes()

    def get_texts(self):
        self._texts = DataJson()[self.widget_id]["texts"]
        return self._texts

    def set_texts(self, value: List[str]):
        if not isinstance(value, list):
            raise TypeError("Argument value type has to be a list.")
        self._texts = value
        DataJson()[self.widget_id]["texts"] = self._texts
        DataJson().send_changes()

    def show_text(self):
        self._show_text = True
        DataJson()[self.widget_id]["show_text"] = self._show_text
        DataJson().send_changes()

    def hide_text(self):
        self._show_text = False
        DataJson()[self.widget_id]["show_text"] = self._show_text
        DataJson().send_changes()

    def get_text_color(self):
        self._text_color = DataJson()[self.widget_id]["text_color"]
        return self._text_color

    def set_text_color(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Argument value type has to be a str.")
        self._text_color = value
        DataJson()[self.widget_id]["text_color"] = self._text_color
        DataJson().send_changes()

    def get_void_color(self):
        self._void_color = DataJson()[self.widget_id]["void_color"]
        return self._void_color

    def set_void_color(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Argument value type has to be a str.")
        self._void_color = value
        DataJson()[self.widget_id]["void_color"] = self._void_color
        DataJson().send_changes()

    def get_disabled_void_color(self):
        self._disabled_void_color = DataJson()[self.widget_id]["disabled_void_color"]
        return self._disabled_void_color

    def set_disabled_void_color(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Argument value type has to be a str.")
        self._disabled_void_color = value
        DataJson()[self.widget_id]["disabled_void_color"] = self._disabled_void_color
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Rate.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            self.get_value()
            func(self._value)

        return _click
