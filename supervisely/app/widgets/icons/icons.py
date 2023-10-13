from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Optional


class Icons(Widget):
    def __init__(
        self,
        class_name: Optional[str] = None,
        color: Optional[str] = None,
        bg_color: Optional[str] = None,
        rounded: Optional[bool] = False,
        image_url: Optional[str] = None,
        widget_id: Optional[str] = None,
    ):
        self._class_name = class_name
        self._color = color
        self._bg_color = bg_color
        self._rounded = rounded
        self._image_url = image_url

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "className": self._class_name,
            "color": self._color,
            "bgColor": self._bg_color,
            "rounded": self._rounded,
            "imageUrl": self._image_url,
        }

    def get_json_state(self):
        return {}

    def set_class_name(self, value: str):
        self._class_name = value
        DataJson()[self.widget_id]["className"] = self._class_name
        DataJson().send_changes()

    def get_class_name(self):
        self._class_name = DataJson()[self.widget_id]["className"]
        return self._class_name

    def set_color(self, value: str):
        self._color = value
        DataJson()[self.widget_id]["color"] = self._color
        DataJson().send_changes()

    def get_color(self):
        self._color = DataJson()[self.widget_id]["color"]
        return self._color

    def set_bg_color(self, value: str):
        self._bg_color = value
        DataJson()[self.widget_id]["bgColor"] = self._bg_color
        DataJson().send_changes()

    def get_bg_color(self):
        self._bg_color = DataJson()[self.widget_id]["bgColor"]
        return self._bg_color

    def set_rounded(self):
        self._rounded = True
        DataJson()[self.widget_id]["rounded"] = self._rounded
        DataJson().send_changes()

    def set_standard(self):
        self._rounded = False
        DataJson()[self.widget_id]["rounded"] = self._rounded
        DataJson().send_changes()

    def set_image_url(self, value: str):
        self._image_url = value
        DataJson()[self.widget_id]["imageUrl"] = self._image_url
        DataJson().send_changes()

    def get_image_url(self):
        self._image_url = DataJson()[self.widget_id]["imageUrl"]
        return self._image_url
