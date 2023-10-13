from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class BindedInputNumber(Widget):
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        min: int = 1,
        max: int = 10000,
        proportional: bool = False,
        widget_id: str = None,
    ):
        self._width = width
        self._height = height
        self._min = min
        self._max = max
        self._proportional = proportional
        self._disabled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "options": {
                "proportions": {
                    "width": self._width,
                    "height": self._height
                }
            },
            "disabled": self._disabled
        }

    def get_json_state(self):
        return {
            "value": {
                "min": self._min,
                "max": self._max,
                "width": self._width,
                "height": self._height,
                "proportional": self._proportional
            }
        }

    @property
    def value(self):
        return self._width, self._height

    @value.setter
    def value(self, width, height):
        self._width = width
        self._height = height
        StateJson()[self.widget_id]["width"] = self._width
        StateJson()[self.widget_id]["height"] = self._height
        StateJson().send_changes()

    def get_value(self):
        width =  StateJson()[self.widget_id]['value']["width"]
        height =  StateJson()[self.widget_id]['value']["height"]
        return width, height

    @property
    def proportional(self):
        return self._proportional

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @min.setter
    def proportional(self, value):
        self._proportional = value
        DataJson()[self.widget_id]["proportional"] = self._proportional
        DataJson().send_changes()

    @min.setter
    def min(self, value):
        self._min = value
        DataJson()[self.widget_id]["min"] = self._min
        DataJson().send_changes()

    @max.setter
    def max(self, value):
        self._max = value
        DataJson()[self.widget_id]["max"] = self._max
        DataJson().send_changes()


    def disable(self):
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()