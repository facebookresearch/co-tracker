from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Sidebar(Widget):
    def __init__(
        self,
        left_content: Widget,
        right_content: Widget,
        width_percent: int = 25,
        widget_id: str = None,
    ):
        super().__init__(widget_id=widget_id, file_path=__file__)
        self._left_content = left_content
        self._right_content = right_content
        self._width_percent = width_percent
        self._options = {"sidebarWidth": self._width_percent}
        StateJson()["app_body_padding"] = "0px"
        StateJson()["menuIndex"] = "1"

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}
