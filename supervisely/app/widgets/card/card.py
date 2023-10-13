from typing import List
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger


class Card(Widget):
    def __init__(
        self,
        title: str = None,
        description: str = None,
        collapsable: bool = False,
        content: Widget = None,
        content_top_right: Widget = None,
        lock_message="Card content is locked",
        widget_id: str = None,
    ):
        self._title = title
        self._description = description
        self._collapsable = collapsable
        self._collapsed = False
        self._content = content
        self._show_slot = False
        self._slot_content = content_top_right
        if self._slot_content is not None:
            self._show_slot = True
        self._options = {"collapsable": self._collapsable, "marginBottom": "0px"}
        self._lock_message = lock_message
        super().__init__(widget_id=widget_id, file_path=__file__)
        self._disabled = {"disabled": False, "message": self._lock_message}

    def get_json_data(self):
        return {
            "title": self._title,
            "description": self._description,
            "collapsable": self._collapsable,
            "options": self._options,
            "show_slot": self._show_slot,
        }

    def get_json_state(self):
        return {"disabled": self._disabled, "collapsed": self._collapsed}

    def collapse(self):
        if self._collapsable is False:
            logger.warn(f"Card {self.widget_id} can not be collapsed")
            return
        self._collapsed = True
        StateJson()[self.widget_id]["collapsed"] = self._collapsed
        StateJson().send_changes()

    def uncollapse(self):
        if self._collapsable is False:
            logger.warn(f"Card {self.widget_id} can not be uncollapsed")
            return
        self._collapsed = False
        StateJson()[self.widget_id]["collapsed"] = self._collapsed
        StateJson().send_changes()

    def lock(self, message: str = None):
        if message is not None:
            self._lock_message = message
        self._disabled = {"disabled": True, "message": self._lock_message}
        StateJson()[self.widget_id]["disabled"] = self._disabled
        StateJson().send_changes()

    def unlock(self):
        self._disabled["disabled"] = False
        StateJson()[self.widget_id]["disabled"] = self._disabled
        StateJson().send_changes()

    def is_locked(self):
        return self._disabled["disabled"]
