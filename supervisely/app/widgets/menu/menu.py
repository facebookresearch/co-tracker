from __future__ import annotations

from typing import List
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


"""

l = sly.app.widgets.Text(text="left part", status="success")
items = [
    sly.app.widgets.Select.Item(label="CPU", value="cpu"),
    sly.app.widgets.Select.Item(label="GPU 0", value="cuda:0"),
    sly.app.widgets.Select.Item(value="option3"),
]
r = sly.app.widgets.Select(items=items, filterable=True, placeholder="select me")

# sidebar = sly.app.widgets.Sidebar(left_pane=l, right_pane=item)

g1_items = [
    sly.app.widgets.Menu.Item(title="m1", content=r),
    sly.app.widgets.Menu.Item(title="m2", content=l),
]
g2_items = [
    sly.app.widgets.Menu.Item(title="m3"),
    sly.app.widgets.Menu.Item(title="m4"),
]
g1 = sly.app.widgets.Menu.Group("g1", g1_items)
g2 = sly.app.widgets.Menu.Group("g2", g2_items)
menu = sly.app.widgets.Menu(groups=[g1, g2])
# menu = sly.app.widgets.Menu(items=g1_items)

"""


class Menu(Widget):
    class Item:
        def __init__(
            self,
            title: str,
            # heading: str = None,
            content: Widget = None,
            index: str = None,
            icon: str = None,
        ) -> Menu.Item:
            self.title = title
            # self.heading = title if heading is None else heading
            self.index = index
            self.content = content
            if index is None:
                self.index = str(self.title)
            self.icon = icon

        def to_json(self):
            return {
                "title": self.title,
                # "heading": self.heading,
                "index": self.index,
                "icon": self.icon,
            }

    class Group:
        def __init__(self, title: str, items: List[Menu.Item] = None) -> Menu.Group:
            self.title = title
            self.items = items

        def to_json(self):
            return {
                "title": self.title,
                "items": [item.to_json() for item in self.items],
            }

    def __init__(
        self,
        items: List[Menu.Item] = None,
        groups: List[Menu.Group] = None,
        index: str = None,
        width_percent: int = 25,
        widget_id: str = None,
    ):
        if items is None and groups is None:
            raise ValueError("One of the arguments has to be defined: items or groups")

        if items is not None and groups is not None:
            raise ValueError(
                "Only one of the arguments has to be defined: items or groups"
            )

        self._items = items
        if self._items is None:
            self._items = []
        self._groups = groups
        if self._groups is None:
            self._groups = []
        self._width_percent = width_percent
        self._options = {"sidebarWidth": self._width_percent}
        self._index = index
        if self._index is None:
            self._index = self._get_default_index()
        else:
            self._validate_index()

        StateJson()["app_body_padding"] = "0px"
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        res = {"items": [], "groups": []}
        if self._items is not None:
            res["items"] = [item.to_json() for item in self._items]
        if self._groups is not None:
            res["groups"] = [g.to_json() for g in self._groups]
        return res

    def _validate_index(self):
        for item in self._items:
            if self._index == item.index:
                return

        for group in self._groups:
            for item in group.items:
                if self._index == item.index:
                    return

        raise ValueError(
            f"Index {self._index} not found, there is no item in menu with such index"
        )

    def _get_default_index(self):
        if len(self._items) > 0:
            return self._items[0].index
        if len(self._groups) > 0:
            return self._groups[0].items[0].index

    def get_json_state(self):
        return {"menuIndex": self._index}
