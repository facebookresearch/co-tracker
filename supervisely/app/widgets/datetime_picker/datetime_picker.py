from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from datetime import datetime
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class DateTimePicker(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: Union[int, str, list, tuple] = None,
        readonly: bool = False,
        disabled: bool = False,
        editable: bool = False,
        clearable: bool = True,
        size: Literal["large", "small", "mini"] = None,
        placeholder: str = "Select date and time",
        w_type: Literal[
            "year", "month", "date", "datetime", "week", "datetimerange", "daterange"
        ] = "datetime",
        format: Literal["yyyy", "MM", "dd", "HH", "mm", "ss"] = "yyyy-MM-dd HH:mm:ss",
        widget_id: str = None,
    ):
        self._value = value
        self._readonly = readonly
        self._disabled = disabled
        self._editable = editable
        self._clearable = clearable
        self._size = size
        self._placeholder = placeholder
        self._w_type = w_type
        self._format = format
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "placeholder": self._placeholder,
            "size": self._size,
            "readonly": self._readonly,
            "disabled": self._disabled,
            "editable": self._editable,
            "clearable": self._clearable,
            "type": self._w_type,
            "format": self._format,
        }

    def get_json_state(self):
        return {"value": self._value}

    def get_value(self):
        if "value" not in StateJson()[self.widget_id].keys():
            return None
        value = StateJson()[self.widget_id]["value"]
        if self._w_type in ["datetimerange", "daterange"] and any(
            [bool(date) is False for date in value]
        ):
            return None
        elif self._w_type not in ["datetimerange", "daterange"] and value == "":
            return None
        return value

    def set_value(self, value: Union[int, str, datetime, list, tuple]):
        if self._w_type in ["year", "month", "date", "datetime", "week"]:
            if type(value) not in [int, str, datetime]:
                raise ValueError(
                    f'Datetime picker type "{self._w_type}" does not support value "{value}" of type: "{str(type(value))}". Value type has to be one of: ["int", "str", "datetime].'
                )
            if isinstance(value, datetime):
                value = str(value)

        if self._w_type in ["datetimerange", "daterange"]:
            if type(value) not in [list, tuple]:
                raise ValueError(
                    f'Datetime picker type "{self._w_type}" does not support value "{value}" of type: "{str(type(value))}". Value type has to be one of: ["list", "tuple"].'
                )
            if len(value) != 2:
                raise ValueError(f"Value length has to be equal 2: {len(value)} != 2")
            value = [str(val) if isinstance(val, datetime) else val for val in value]

        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(DateTimePicker.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._value = res
            func(res)

        return _click
