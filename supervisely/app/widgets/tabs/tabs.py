from typing import List, Optional, Dict
from supervisely.app import StateJson
from supervisely.app.widgets import Widget
import traceback
from supervisely import logger


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Tabs(Widget):
    class Routes:
        CLICK = "tab_clicked_cb"

    class TabPane:
        def __init__(self, label: str, content: Widget):
            self.label = label
            self.name = label  # identifier corresponding to the active tab
            self.content = content

    def __init__(
        self,
        labels: List[str],
        contents: List[Widget],
        type: Optional[Literal["card", "border-card"]] = "border-card",
        widget_id=None,
    ):
        if len(labels) != len(contents):
            raise ValueError("labels length must be equal to contents length in Tabs widget.")
        if len(labels) > 10:
            raise ValueError("You can specify up to 10 tabs.")
        if len(set(labels)) != len(labels):
            raise ValueError("All of tab labels should be unique.")
        self._items = []
        for label, widget in zip(labels, contents):
            self._items.append(Tabs.TabPane(label=label, content=widget))
        self._value = labels[0]
        self._type = type
        self._click_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"type": self._type}

    def get_json_state(self) -> Dict:
        return {"value": self._value}

    def set_active_tab(self, value: str):
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_active_tab(self) -> str:
        return StateJson()[self.widget_id]["value"]

    def click(self, func):
        route_path = self.get_route_path(Tabs.Routes.CLICK)
        server = self._sly_app.get_server()

        self._click_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_active_tab()
            if res is not None:
                try:
                    return func(res)
                except Exception as e:
                    logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                    raise e

        return _click
