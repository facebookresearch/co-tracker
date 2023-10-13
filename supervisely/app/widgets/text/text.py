from typing import Optional
from supervisely.app import DataJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

TEXT = "text"
INFO = "info"
SUCCESS = "success"
WARNING = "warning"
ERROR = "error"

type_to_icon = {
    TEXT: "",
    INFO: "zmdi zmdi-info",
    SUCCESS: "zmdi zmdi-check-circle",
    WARNING: "zmdi zmdi-alert-triangle",
    ERROR: "zmdi zmdi-alert-circle",
}

type_to_icon_color = {
    TEXT: "#000000",
    INFO: "#3B96FF",
    SUCCESS: "#13ce66",
    WARNING: "#ffa500",
    ERROR: "#ff0000",
}

type_to_text_color = {
    TEXT: "#000000",
    INFO: "#5a6772",
    SUCCESS: "#5a6772",
    WARNING: "#5a6772",
    ERROR: "#5a6772",
}


class Text(Widget):
    def __init__(
        self,
        text: str = None,
        status: Literal["text", "info", "success", "warning", "error"] = "text",
        color: Optional[str] = None,
        widget_id: str = None,
    ):
        self._text = None
        self._status = None
        self._icon = None
        self._icon_color = None
        self._text_color = None
        super().__init__(widget_id=widget_id, file_path=__file__)
        self.set(text, status)
        if color is not None:
            self.color = color

    def get_json_data(self):
        return {
            "status": self._status,
            "text": self._text,
            "text_color": self._text_color,
            "icon": self._icon,
            "icon_color": self._icon_color,
        }

    def get_json_state(self):
        return None

    @property
    def text(self):
        return self._text
    
    def get_value(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        self.update_data()
        DataJson().send_changes()

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value: Literal["text", "info", "success", "warning", "error"]):
        if value not in type_to_icon:
            raise ValueError(f'Unknown status "{value}"')
        self._status = value
        self._icon = type_to_icon[self._status]
        self._icon_color = type_to_icon_color[self._status]
        self._text_color = type_to_text_color[self._status]
        self.update_data()
        DataJson().send_changes()

    def set(self, text: str, status: Literal["text", "info", "success", "warning", "error"]):
        self.text = text
        self.status = status

    @property
    def color(self):
        return self._text_color
    
    @color.setter
    def color(self, value):
        self._text_color = value
        self.update_data()
        DataJson().send_changes()
