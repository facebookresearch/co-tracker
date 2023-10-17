from typing import List, Optional, Dict, Union, Literal
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Tooltip(Widget):
    def __init__(
        self,
        text: Union[str, List[str]],
        content: Widget,
        color_theme: Optional[Literal["dark", "light"]] = "dark",
        placement: Optional[
            Literal[
                "top",
                "top-start",
                "top-end",
                "bottom",
                "bottom-start",
                "bottom-end",
                "left",
                "left-start",
                "left-end",
                "right",
                "right-start",
                "right-end",
            ]
        ] = "bottom",
        offset: Optional[int] = 0,
        transition: Optional[
            Literal[
                "el-fade-in-linear",
                "el-fade-in",
            ]
        ] = "el-fade-in-linear",
        visible_arrow: Optional[bool] = True,
        open_delay: Optional[int] = 0,
        enterable: Optional[bool] = True,
        hide_after: Optional[int] = 0,
        widget_id: Optional[str] = None,
    ):
        self._text = text
        self._content = content
        self._color_theme = color_theme
        self._placement = placement
        self._offset = offset
        self._transition = transition
        self._visible_arrow = visible_arrow
        self._open_delay = open_delay
        self._enterable = enterable
        self._hide_after = hide_after
        self._multiline = True if isinstance(self._text, List) else False

        if open_delay >= hide_after and hide_after != 0:
            raise ValueError("The value 'open_delay' must be less than 'hide_after'")

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "text": self._text,
            "color_theme": self._color_theme,
            "placement": self._placement,
            "offset": self._offset,
            "transition": self._transition,
            "visible_arrow": self._visible_arrow,
            "open_delay": self._open_delay,
            "enterable": self._enterable,
            "hide_after": self._hide_after,
            "multiline": self._multiline,
        }

    def get_json_state(self):
        return None

    def set_text(self, text: Union[str, List[str]]):
        """
        To make tooltip multiline - pass text as list of lines
        """
        self._text = text
        self._multiline = True if isinstance(self._text, List) else False
        DataJson()[self.widget_id]["text"] = self._text
        DataJson()[self.widget_id]["multiline"] = self._multiline
        DataJson().send_changes()

    def set_placement(
        self,
        placement: Literal[
            "top",
            "top-start",
            "top-end",
            "bottom",
            "bottom-start",
            "bottom-end",
            "left",
            "left-start",
            "left-end",
            "right",
            "right-start",
            "right-end",
        ],
    ):
        self._placement = placement
        DataJson()[self.widget_id]["placement"] = self._placement
        DataJson().send_changes()

    def set_offset(self, offset: int):
        self._offset = offset
        DataJson()[self.widget_id]["offset"] = self._offset
        DataJson().send_changes()

    def set_transition(
        self,
        transition: Literal["el-fade-in-linear", "el-fade-in"],
    ):
        self._transition = transition
        DataJson()[self.widget_id]["transition"] = self._transition
        DataJson().send_changes()

    def set_arrow_visibility(self, visible_arrow: bool):
        self._visible_arrow = visible_arrow
        DataJson()[self.widget_id]["visible_arrow"] = self._visible_arrow
        DataJson().send_changes()

    def set_open_delay(self, open_delay: int):
        """
        Milliseconds
        """
        if open_delay >= self._hide_after and self._hide_after != 0:
            raise ValueError(
                f"The value 'open_delay: {open_delay}' must be less than 'hide_after: {self._hide_after}'"
            )
        else:
            self._open_delay = open_delay
            DataJson()[self.widget_id]["open_delay"] = self._open_delay
            DataJson().send_changes()

    def set_hide_after(self, hide_after: int):
        """
        Milliseconds
        """
        if self._open_delay >= hide_after and hide_after != 0:
            raise ValueError(
                f"The value 'hide_after: {hide_after}' must be greater than 'open_delay: {self._open_delay}'"
            )
        else:
            self._hide_after = hide_after
            DataJson()[self.widget_id]["hide_after"] = self._hide_after
            DataJson().send_changes()
