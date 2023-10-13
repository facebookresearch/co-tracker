from supervisely.app import DataJson
from supervisely.app.widgets import Widget

INFO = "info"
WARNING = "warning"
ERROR = "error"


class DoneLabel(Widget):
    def __init__(
        self,
        text: str = None,
        widget_id: str = None,
    ):
        self._text = text
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {"text": self._text}

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
