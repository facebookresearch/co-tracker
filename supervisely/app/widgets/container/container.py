from typing import List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Container(Widget):
    def __init__(
        self,
        widgets: List[Widget] = [],
        direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 10,
        fractions: List[int] = None,
        overflow: Optional[Literal["scroll", "wrap"]] = "scroll",
        style: str = "",
        widget_id: str = None,
    ):
        self._widgets = widgets
        self._direction = direction
        self._gap = gap
        self._overflow = overflow
        self._style = style
        
        if self._overflow not in ["scroll", "wrap", None]:
            raise ValueError("overflow can be only 'scroll', 'wrap' or None")
        
        if self._direction == "vertical" and self._overflow == "wrap":
            raise ValueError("overflow can be 'wrap' only with horizontal direction")
        
        if self._direction == "vertical" and fractions is not None:
            raise ValueError("fractions can be defined only with horizontal direction")

        if fractions is not None and len(widgets) != len(fractions):
            raise ValueError(
                "len(widgets) != len(fractions): fractions have to be defined for all widgets"
            )

        if self._direction == "vertical":
            self._overflow = None

        self._fractions = fractions
        self._flex_direction = "column"
        if direction == "horizontal":
            self._flex_direction = "row"
            if self._fractions is None:
                self._fractions = ["1 1 auto"] * len(self._widgets)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None
