from supervisely.app.widgets import Widget
from typing import Dict, List


"""
<div 
    {% if not loop.last %}
    style="margin-bottom: {{{widget._gap}}}px;"
    {% endif %}
>
    {{{w}}}
</div>
"""


class Flexbox(Widget):
    # https://www.w3schools.com/css/css3_flexbox.asp
    def __init__(
        self,
        widgets: List[Widget],
        gap: int = 10,
        center_content: bool = False,
        widget_id: str = None,
    ):
        self._widgets = widgets
        self._gap = gap
        self._center_content = center_content
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {"center": self._center_content}
        return res

    def get_json_state(self) -> Dict:
        return {}
