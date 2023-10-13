try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import DataJson
from supervisely.app.widgets import Widget

SUCCESS = "success"
INFO = "info"
WARNING = "warning"
ERROR = "error"

BOXTYPE2ICON = {
    SUCCESS: "zmdi-check-circle",
    INFO: "zmdi-info",
    WARNING: "zmdi-alert-triangle",
    ERROR: "zmdi-alert-circle",
}


class NotificationBox(Widget):
    def __init__(
        self,
        title: str = None,
        description: str = None,
        box_type: Literal["success", "info", "warning", "error"] = INFO,
        widget_id: str = None,
    ):
        self._title = title
        self._description = description
        # if self._title is None and self._description is None:
        #     raise ValueError(
        #         "Both title and description can not be None at the same time"
        #     )

        self.box_type = box_type
        if self.box_type not in [SUCCESS, INFO, WARNING, ERROR]:
            raise ValueError(
                f"Box type {box_type} type isn't supported. Please select one of {[SUCCESS, INFO, WARNING, ERROR]} box_type"
            )

        self.icon = BOXTYPE2ICON[self.box_type]

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "title": self._title,
            "description": self._description,
            "icon": self.icon,
        }

    def get_json_state(self):
        return None

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        DataJson()[self.widget_id]["title"] = self._title
        DataJson().send_changes()

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value
        DataJson()[self.widget_id]["description"] = self._description
        DataJson().send_changes()

    def set(self, title: str, description: str):
        self.title = title
        self.description = description
