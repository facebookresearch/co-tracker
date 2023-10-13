from supervisely.app.widgets import Widget, ConditionalWidget
from typing import Dict


class OneOf(Widget):
    def __init__(
        self,
        conditional_widget: ConditionalWidget,
        widget_id: str = None,
    ):
        self._conditional_widget = conditional_widget
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {}
