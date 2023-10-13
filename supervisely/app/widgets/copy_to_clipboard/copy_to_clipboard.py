from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Dict, Optional, Union
from supervisely.app.widgets import Editor, Text, TextArea, Input


class CopyToClipboard(Widget):
    def __init__(
        self,
        content: Union[Editor, Text, TextArea, Input, str] = "",
        widget_id: Optional[str] = None,
    ):
        self._content = content

        if not isinstance(content, (str, Editor, Text, TextArea, Input)): 
            raise TypeError(
                f"Supported types: str, Editor, Text, TextArea, Input. Your type: {type(content).__name__}"
            )
        if isinstance(content, str):
            self._content_widget_type = 'str'
            self._curr_prop_name = None
            self._content_value = content
        else:
            if isinstance(content, (Editor, Input)):
                self._content_widget_type = 'input'
                self._curr_prop_name = "value" if isinstance(content, Input) else "text"
            elif isinstance(content, (Text, TextArea)):
                self._content_widget_type = 'text'
                self._curr_prop_name = "value" if isinstance(content, TextArea) else "text"
            self._content_value = content.get_value()

        super().__init__(widget_id=widget_id, file_path=__file__)


    def get_json_data(self) -> Dict:
        return {
            "content": self._content_value,
            "curr_property": self._curr_prop_name
        }

    def get_json_state(self) -> Dict:
        return {
            "content": self._content_value,
            "curr_property": self._curr_prop_name
        }

    def get_content(self) -> Union[Editor, Input, Text, TextArea , str]:
        return self._content
