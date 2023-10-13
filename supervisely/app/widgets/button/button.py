from functools import wraps

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Button(Widget):
    class Routes:
        CLICK = "button_clicked_cb"

    def __init__(
        self,
        text: str = "Button",
        button_type: Literal["primary", "info", "warning", "danger", "success", "text"] = "primary",
        button_size: Literal["mini", "small", "large"] = None,
        plain: bool = False,
        show_loading: bool = True,
        icon: str = None,  # for example "zmdi zmdi-play" from http://zavoloklom.github.io/material-design-iconic-font/icons.html
        icon_gap: int = 5,
        widget_id=None,
        link: str = None,
    ):
        self._widget_routes = {}

        self._text = text
        self._button_type = button_type
        self._button_size = button_size
        self._plain = plain
        self._icon_gap = icon_gap
        self._link = link
        if icon is None:
            self._icon = ""
        else:
            self._icon = f'<i class="{icon}" style="margin-right: {icon_gap}px"></i>'

        self._loading = False
        self._disabled = False
        self._show_loading = show_loading
        self._click_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "text": self._text,
            "button_type": self._button_type,
            "plain": self._plain,
            "button_size": self._button_size,
            "loading": self._loading,
            "disabled": self._disabled,
            "icon": self._icon,
            "link": self._link,
        }

    def get_json_state(self):
        return None

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        DataJson()[self.widget_id]["text"] = self._text
        DataJson().send_changes()

    @property
    def icon(self):
        return self._icon

    @icon.setter
    def icon(self, value):
        if value is None:
            self._icon = ""
        else:
            self._icon = f'<i class="{value}" style="margin-right: {self._icon_gap}px"></i>'
        DataJson()[self.widget_id]["icon"] = self._icon
        DataJson().send_changes()

    @property
    def button_type(self):
        return self._button_type

    @button_type.setter
    def button_type(self, value):
        self._button_type = value
        DataJson()[self.widget_id]["button_type"] = self._button_type
        DataJson().send_changes()

    @property
    def plain(self):
        return self._plain

    @plain.setter
    def plain(self, value):
        self._plain = value
        DataJson()[self.widget_id]["plain"] = self._plain
        DataJson().send_changes()

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, value):
        self._link = value
        DataJson()[self.widget_id]["link"] = self._link
        DataJson().send_changes()

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    @property
    def show_loading(self):
        return self._show_loading

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value):
        self._disabled = value
        DataJson()[self.widget_id]["disabled"] = self._disabled

    def click(self, func):
        # from fastapi import Request

        route_path = self.get_route_path(Button.Routes.CLICK)
        server = self._sly_app.get_server()
        self._click_handled = True

        @server.post(route_path)
        def _click():
            # maybe work with headers and store some values there r: Request
            if self.show_loading:
                self.loading = True
            try:
                func()
            except Exception as e:
                if self.show_loading and self.loading:
                    self.loading = False
                raise e
            if self.show_loading:
                self.loading = False

        return _click
