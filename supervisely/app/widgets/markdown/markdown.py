from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from typing import Dict, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Markdown(Widget):
    def __init__(
        self,
        content: str = "",
        height: Union[int, Literal["fit-content"]] = "fit-content",
        widget_id: str = None,
    ):
        self._content = content
        self._height = f"{height}px" if type(height) == int else height

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "content": self._content,
            "options": {"height": self._height},
        }

    def get_json_state(self) -> Dict:
        return {}

    def set_content(self, content: str) -> None:
        if not isinstance(content, str):
            raise TypeError("Content type has to be str.")
        self._content = content
        DataJson()[self.widget_id]["content"] = self._content
        DataJson().send_changes()

    def get_content(self) -> str:
        self._content = DataJson()[self.widget_id]["content"]
        return self._content

    def get_height(self) -> Union[int, Literal["fit-content"]]:
        self._height = DataJson()[self.widget_id]["options"]["height"]
        if self._height == "fit-content":
            return self._height
        return int(self._height[:-2])

    def set_height(self, height: Union[int, Literal["fit-content"]]) -> None:
        if type(height) != int and height != "fit-content":
            raise TypeError("Height value type has to be an integer or 'fit-content' string.")
        self._height = f"{height}px" if type(height) == int else height
        DataJson()[self.widget_id]["options"]["height"] = self._height
        DataJson().send_changes()
