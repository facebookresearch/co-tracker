from __future__ import annotations
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget, ConditionalWidget
from typing import List, Dict, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# r = sly.app.widgets.Text(text="right part", status="error")
# items = [
#     sly.app.widgets.Select.Item(value="option1"),
#     sly.app.widgets.Select.Item(value="option2"),
#     sly.app.widgets.Select.Item(value="option3"),
# ]
# r = sly.app.widgets.Select(items=items, filterable=True, placeholder="select me")


# groups = [
#     sly.app.widgets.Select.Group(
#         label="group1",
#         items=[
#             sly.app.widgets.Select.Item(value="g1-option1"),
#             sly.app.widgets.Select.Item(value="g1-option2"),
#         ],
#     ),
#     sly.app.widgets.Select.Group(
#         label="group2",
#         items=[
#             sly.app.widgets.Select.Item(value="g2-option1"),
#             sly.app.widgets.Select.Item(value="g2-option2"),
#         ],
#     ),
# ]
# r = sly.app.widgets.Select(groups=groups, filterable=True, placeholder="select me")


# @r.value_changed
# def do(value):
#     print(f"new value is: {value}")


class Select(ConditionalWidget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(
            self,
            value,
            label: str = None,
            content: Widget = None,
            right_text: str = None,
        ) -> Select.Item:
            self.value = value
            self.label = label
            if label is None:
                self.label = str(self.value)
            self.content = content
            self.right_text = right_text

        def to_json(self):
            return {"label": self.label, "value": self.value, "right_text": self.right_text}

    class Group:
        def __init__(self, label, items: List[Select.Item] = None) -> Select.Item:
            self.label = label
            self.items = items

        def to_json(self):
            res = {
                "label": self.label,
                "options": [item.to_json() for item in self.items],
            }
            return res

    def __init__(
        self,
        items: List[Select.Item] = None,
        groups: List[Select.Group] = None,
        filterable: bool = False,
        placeholder: str = "select",
        size: Literal["large", "small", "mini"] = None,
        multiple: bool = False,
        widget_id: str = None,
        items_links: List[str] = None,
    ) -> Select:
        if items is None and groups is None:
            raise ValueError("One of the arguments has to be defined: items or groups")

        if items is not None and groups is not None:
            raise ValueError("Only one of the arguments has to be defined: items or groups")

        self._groups = groups
        self._filterable = filterable
        self._placeholder = placeholder
        self._changes_handled = False
        self._size = size
        self._multiple = multiple
        self._with_link = False
        self._links = None
        if items_links is not None:
            if items is None:
                raise ValueError("links are not supported when groups are provided to Select")
            else:
                assert len(items_links) == len(items)
            self._with_link = True
            self._links = {items[i].value: link for i, link in enumerate(items_links)}

        super().__init__(items=items, widget_id=widget_id, file_path=__file__)

    def _get_first_value(self) -> Select.Item:
        if self._items is not None and len(self._items) > 0:
            return self._items[0]
        if self._groups is not None and len(self._groups) > 0 and len(self._groups[0].items) > 0:
            return self._groups[0].items[0]
        return None

    def get_json_data(self) -> Dict:
        res = {
            "filterable": self._filterable,
            "placeholder": self._placeholder,
            "multiple": self._multiple,
            "items": None,
            "groups": None,
            "with_link": self._with_link,
        }
        if self._items is not None:
            res["items"] = [item.to_json() for item in self._items]
        if self._groups is not None:
            res["groups"] = [group.to_json() for group in self._groups]
        if self._size is not None:
            res["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        first_item = self._get_first_value()
        value = None
        if first_item is not None:
            value = first_item.value
        return {"value": value, "links": self._links}

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def get_label(self):
        for item in self.get_items():
            if item.value == self.get_value():
                return item.label

    def value_changed(self, func):
        route_path = self.get_route_path(Select.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        async def _click():
            res = self.get_value()
            func(res)

        return _click

    def get_items(self) -> List[Select.Item]:
        res = []
        if self._items is not None:
            res.extend(self._items)
        if self._groups is not None:
            for group in self._groups:
                res.extend(group.items)
        return res

    def set(
        self,
        items: List[Select.Item] = None,
        groups: List[Select.Group] = None,
    ):
        if items is None and groups is None:
            raise ValueError("One of the arguments has to be defined: items or groups")
        if items is not None and groups is not None:
            raise ValueError("Only one of the arguments has to be defined: items or groups")

        self._items = items
        self._groups = groups

        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()

    def set_value(self, value):
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def disable_item(self, item_index, group_index=None):
        if group_index is None:
            DataJson()[self.widget_id]["items"][item_index].update({"disabled": True})
        else:
            DataJson()[self.widget_id]["groups"][group_index]["options"][item_index].update(
                {"disabled": True}
            )
        DataJson().send_changes()

    def enable_item(self, item_index, group_index=None):
        if group_index is None:
            DataJson()[self.widget_id]["items"][item_index].update({"disabled": False})
        else:
            DataJson()[self.widget_id]["groups"][group_index]["options"][item_index].update(
                {"disabled": False}
            )
        DataJson().send_changes()

    def disable_group(self, group_index):
        DataJson()[self.widget_id]["groups"][group_index].update({"disabled": True})
        DataJson().send_changes()

    def enable_group(self, group_index):
        DataJson()[self.widget_id]["groups"][group_index].update({"disabled": False})
        DataJson().send_changes()


class SelectString(Select):
    def __init__(
        self,
        values: List[str],
        labels: Optional[List[str]] = None,
        filterable: Optional[bool] = False,
        placeholder: Optional[str] = "select",
        size: Optional[Literal["large", "small", "mini"]] = None,
        multiple: Optional[bool] = False,
        widget_id: Optional[str] = None,
        items_right_text: List[str] = None,
        items_links: List[str] = None,
    ):
        right_text = [None] * len(values)
        if items_right_text is not None:
            if len(values) != len(items_right_text):
                raise ValueError("items_right_text length must be equal to values length.")
            right_text = items_right_text

        if labels is not None:
            if len(values) != len(labels):
                raise ValueError("values length must be equal to labels length.")
            items = []
            for value, label, rtext in zip(values, labels, right_text):
                items.append(Select.Item(value, label, right_text=rtext))
        else:
            items = [
                Select.Item(value, right_text=rtext) for value, rtext in zip(values, right_text)
            ]

        super(SelectString, self).__init__(
            items=items,
            groups=None,
            filterable=filterable,
            placeholder=placeholder,
            multiple=multiple,
            size=size,
            widget_id=widget_id,
            items_links=items_links,
        )

    def _get_first_value(self) -> Select.Item:
        if self._items is not None and len(self._items) > 0:
            return self._items[0]
        return None

    def get_items(self) -> List[str]:
        return [item.value for item in self._items]

    def set(
        self,
        values: List[str],
        labels: Optional[List[str]] = None,
        right_text: Optional[List[str]] = None,
        items_links: Optional[List[str]] = None,
    ):
        right_texts = [None] * len(values)
        if right_text is not None:
            if len(values) != len(right_text):
                raise ValueError("right_text length must be equal to values length.")
            right_texts = right_text

        if labels is not None:
            if len(values) != len(labels):
                raise ValueError("values length must be equal to labels length.")
            self._items = []
            for value, label, rtext in zip(values, labels, right_texts):
                self._items.append(Select.Item(value, label, right_text=rtext))
        else:
            self._items = [
                Select.Item(value, right_text=rtext) for value, rtext in zip(values, right_texts)
            ]
        if items_links is not None:
            assert len(items_links) == len(values)
            self._with_link = True
            self._links = {value: items_links[i] for i, value in enumerate(values)}
        else:
            self._with_link = False
            self._links = None
        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()
