from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, Button
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Editor(Widget):
    def __init__(
        self,
        initial_text: Optional[str] = "",
        height_px: Optional[int] = 100,
        height_lines: Optional[int] = None,  # overwrites height_px if specified. If >= 1000, all lines will be displayed.
        language_mode: Optional[Literal["json", "html", "plain_text", "yaml", "python"]] = "json",
        readonly: Optional[bool] = False,
        show_line_numbers: Optional[bool] = True,
        highlight_active_line: Optional[bool] = True,
        restore_default_button: Optional[bool] = True,
        widget_id: Optional[int] = None,
    ):
        self._initial_code = initial_text
        self._current_code = initial_text
        self._height_px = height_px
        self._height_lines = height_lines
        self._language_mode = language_mode
        self._readonly = readonly
        self._show_line_numbers = show_line_numbers
        self._highlight_active_line = highlight_active_line
        self._restore_button = None
        if restore_default_button:
            self._restore_button = Button("Restore Default", button_type="text", plain=True)

            @self._restore_button.click
            def restore_default():
                self._current_code = self._initial_code
                StateJson()[self.widget_id]["text"] = self._current_code
                StateJson().send_changes()

        super(Editor, self).__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "editor_options": {
                "height": f"{self._height_px}px",
                "mode": f"ace/mode/{self._language_mode}",
                "readOnly": self._readonly,
                "showGutter": self._show_line_numbers,
                "maxLines": self._height_lines,
                "highlightActiveLine": self._highlight_active_line,
            },
        }

    def get_json_state(self):
        return {"text": self._current_code}

    def get_text(self) -> str:
        return StateJson()[self.widget_id]["text"]
    
    def get_value(self) -> str:
        return StateJson()[self.widget_id]["text"]

    def set_text(
        self,
        text: Optional[str] = "",
        language_mode: Optional[Literal["json", "html", "plain_text", "yaml", "python"]] = None,
    ) -> None:
        self._initial_code = text
        self._current_code = text
        self._language_mode = language_mode
        StateJson()[self.widget_id]["text"] = text
        StateJson().send_changes()
        if language_mode is not None:
            self._language_mode = f"ace/mode/{language_mode}"
            DataJson()[self.widget_id]["editor_options"]["mode"] = self._language_mode
            DataJson().send_changes()

    @property
    def readonly(self) -> bool:
        return self._readonly

    @readonly.setter
    def readonly(self, value: bool):
        self._readonly = value
        DataJson()[self.widget_id]["editor_options"]["readOnly"] = self._readonly
        DataJson().send_changes()

    def show_line_numbers(self):
        self._show_line_numbers = True
        DataJson()[self.widget_id]["editor_options"]["showGutter"] = self._show_line_numbers
        DataJson().send_changes()

    def hide_line_numbers(self):
        self._show_line_numbers = False
        DataJson()[self.widget_id]["editor_options"]["showGutter"] = self._show_line_numbers
        DataJson().send_changes()
