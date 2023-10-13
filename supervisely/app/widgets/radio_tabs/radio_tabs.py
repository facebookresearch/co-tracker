from typing import List, Optional, Dict
from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class RadioTabs(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed_cb"

    class RadioTabPane:
        def __init__(
            self,
            title: str,
            content: Widget,
            subtitle: Optional[str] = "",
        ):
            self.title = title
            self.subtitle = subtitle
            self.name = title  # identifier corresponding to the active tab
            self.content = content

    def __init__(
        self,
        titles: List[str],
        contents: List[Widget],
        descriptions: Optional[List[str]] = None,
        widget_id=None,
    ):
        if len(titles) != len(contents):
            raise ValueError(
                "titles length must be equal to contents length in RadioTabs widget."
            )
        if len(titles) > 10:
            raise ValueError("You can specify up to 10 tabs.")
        if descriptions is None:
            descriptions = [""] * len(titles)
        else:
            if len(titles) != len(descriptions):
                raise ValueError(
                    "descriptions length must be equal to titles length in RadioTabs widget."
                )
        if len(set(titles)) != len(titles):
            raise ValueError("All of tab labels should be unique.")
        self._items = []
        for title, widget, description in zip(titles, contents, descriptions):
            self._items.append(
                RadioTabs.RadioTabPane(
                    title=title, content=widget, subtitle=description
                )
            )
        self._value = titles[0]
        self._changes_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def value_changed(self, func):
        route_path = self.get_route_path(RadioTabs.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_active_tab()
            func(res)

        return _value_changed

    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {"value": self._value}

    def set_active_tab(self, value: str):
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_active_tab(self) -> str:
        return StateJson()[self.widget_id]["value"]
