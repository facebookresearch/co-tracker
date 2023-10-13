from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Union, List


class Slider(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
            self,
            value: Union[int, List[int]] = 0,
            min: int = 0,
            max: int = 100,
            step: int = 1,
            show_input: bool = False,
            show_input_controls: bool = False,
            show_stops: bool = False,
            show_tooltip: bool = True,
            range: bool = False,  # requires value to be List[int, int]
            vertical: bool = False,
            height: int = None,
            widget_id: str = None
    ):
        self._value = value
        self._min = min
        self._max = max
        self._step = step
        self._show_input = show_input
        self._show_input_controls = show_input_controls if show_input else False
        self._show_stops = show_stops
        self._show_tooltip = show_tooltip
        self._range = False if show_input else range
        self._vertical = vertical
        self._height = f"{height}px" if vertical else None
        self._changes_handled = False

        self._validate_min_max(min, max)
        self._validate_default_value(value)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _validate_min_max(self, min_val: int, max_val: int):
        if min_val > max_val:
            raise ValueError(f"Minimum value: '{min_val}' can't be bigger than maximum value: '{max_val}'")

    def _validate_default_value(self, value: Union[int, List[int]]):
        if isinstance(value, int):
            if self._range:
                raise ValueError(f"value = '{value}', should be 'list' if range is True")
            if value < self._min:
                self._value = self._min
            elif value > self._max:
                self._value = self._max
        elif isinstance(value, list):
            if not self._range:
                raise ValueError(f"value = '{value}', should be 'int' if range is False")
            if value[0] < self._min and value[1] > self._max:
                self._value = self._min
                self._value = self._max
            elif value[0] < self._min and value[1] < self._min:
                self._value[0] = self._min
                self._value[1] = self._min
            elif value[0] > self._max and value[1] > self._max:
                self._value[0] = self._max
                self._value[1] = self._max
        else:
            raise ValueError(f"value = '{value}', should be 'int' or 'list', not '{type(value)}'")

    def get_json_data(self):
        return {
            "min": self._min,
            "max": self._max,
            "step": self._step,
            "showInput": self._show_input,
            "showInputControls": self._show_input_controls,
            "showStops": self._show_stops,
            "showTooltip": self._show_tooltip,
            "range": self._range,
            "vertical": self._vertical,
            "height": self._height
        }

    def get_json_state(self):
        return {"value": self._value}

    def set_value(self, value: Union[int, List[int]]):
        self._validate_default_value(value)
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def set_min(self, value: int):
        self._validate_min_max(value, self.get_max())
        DataJson()[self.widget_id]["min"] = value
        DataJson().send_changes()

    def get_min(self):
        return DataJson()[self.widget_id]["min"]

    def set_max(self, value: int):
        self._validate_min_max(self.get_min(), value)
        DataJson()[self.widget_id]["max"] = value
        DataJson().send_changes()

    def get_max(self):
        return DataJson()[self.widget_id]["max"]

    def set_step(self, value: int):
        DataJson()[self.widget_id]["step"] = value
        DataJson().send_changes()

    def get_step(self):
        return DataJson()[self.widget_id]["step"]

    def is_input_enabled(self):
        return DataJson()[self.widget_id]["showInput"]

    def show_input(self):
        DataJson()[self.widget_id]["showInput"] = True
        DataJson().send_changes()

    def hide_input(self):
        DataJson()[self.widget_id]["showInput"] = False
        DataJson().send_changes()

    def is_input_controls_enabled(self):
        return DataJson()[self.widget_id]["showInputControls"]

    def show_input_controls(self):
        DataJson()[self.widget_id]["showInputControls"] = True
        DataJson().send_changes()

    def hide_input_controls(self):
        DataJson()[self.widget_id]["showInputControls"] = False
        DataJson().send_changes()

    def is_step_enabled(self):
        return DataJson()[self.widget_id]["showStops"]

    def show_steps(self):
        DataJson()[self.widget_id]["showStops"] = True
        DataJson().send_changes()

    def hide_steps(self):
        DataJson()[self.widget_id]["showStops"] = False
        DataJson().send_changes()

    def is_tooltip_enabled(self):
        return DataJson()[self.widget_id]["showTooltip"]

    def show_tooltip(self):
        DataJson()[self.widget_id]["showTooltip"] = True
        DataJson().send_changes()

    def hide_tooltip(self):
        DataJson()[self.widget_id]["showTooltip"] = False
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Slider.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._value = res
            func(res)

        return _click
