from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Set, Union, Dict, Any, Optional


class Collapse(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item(object):
        def __init__(self, name: str, title: str, content: Union[Widget, str]) -> None:
            self.name = name  # unique identification of the panel
            self.title = title
            self.content = content

        def to_json(self) -> Dict[str, Any]:
            if isinstance(self.content, str):
                content_type = "text"
            else:
                content_type = str(type(self.content))
            return {
                "name": self.name,
                "label": self.title,
                "content_type": content_type,
            }

    def __init__(
        self,
        items: Optional[List[Collapse.Item]] = None,
        accordion: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        if items is None:
            items = [Collapse.Item("default", "Empty default item", "")]

        labels = [item.name for item in items]
        if len(set(labels)) != len(labels):
            raise ValueError("All items must have a unique name.")

        self._items: List[Collapse.Item] = items

        self._accordion = accordion
        self._active_panels = []

        self._items_names = set(labels)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_items_json(self) -> List[Dict[str, Any]]:
        return [item.to_json() for item in self._items]

    def get_json_data(self):
        return {
            "accordion": self._accordion,
            "items": self._get_items_json(),
        }

    def get_json_state(self):
        return {"value": self._active_panels}

    def set_active_panel(self, value: Union[str, List[str]]):
        """Set active panel or panels.

        :param value: panel name(s)
        :type value: Union[str, List[str]]
        :raises TypeError: value of type List[str] can't be setted, if accordion is True.
        :raises ValueError: panel with such title doesn't exist.
        """
        if isinstance(value, list):
            if self._accordion:
                raise TypeError(
                    "Only one panel could be active in accordion mode. Use `str`, not `list`."
                )
            for name in value:
                if name not in self._items_names:
                    raise ValueError(
                        f"Can't activate panel `{name}`: item with such name doesn't exist."
                    )
        else:
            if value not in self._items_names:
                raise ValueError(
                    f"Can't activate panel `{value}`: item with such name doesn't exist."
                )

        if isinstance(value, str):
            self._active_panels = [value]
        else:
            self._active_panels = value

        StateJson()[self.widget_id]["value"] = self._active_panels
        StateJson().send_changes()

    def get_active_panel(self) -> Union[str, List[str]]:
        return StateJson()[self.widget_id]["value"]

    def get_items(self):
        return DataJson()[self.widget_id]["items"]

    def set_items(self, value: List[Collapse.Item]):
        names = [val.name for val in value]

        self._items_names = self._make_set_from_unique(names)
        self._items = value
        self._active_panels = []

        DataJson()[self.widget_id]["items"] = self._get_items_json()
        DataJson().send_changes()

    def add_items(self, value: List[Collapse.Item]):
        names = [val.name for val in value]
        set_of_names = self._make_set_from_unique(names)

        for name in names:
            if name in self._items_names:
                raise ValueError(f"Item with name {name} already exists.")

        self._items.extend(value)
        self._items_names.update(set_of_names)
        DataJson()[self.widget_id]["items"] = self._get_items_json()
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Collapse.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            active = self.get_active_panel()
            self._active_panels = active
            func(active)

        return _click

    @property
    def items_names(self):
        return self._items_names

    def _make_set_from_unique(self, names: List[str]) -> Set[str]:
        set_of_names = set(names)
        if len(names) != len(set_of_names):
            raise ValueError("All items must have a unique name.")
        return set_of_names
