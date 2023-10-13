from typing import Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.app.widgets.select_sly_utils import _get_int_or_env


class SelectTeam(Widget):
    def __init__(
        self,
        default_id: int = None,
        show_label: bool = True,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
    ):
        self._api = Api()
        self._default_id = default_id
        self._show_label = show_label
        self._size = size

        self._default_id = _get_int_or_env(self._default_id, "context.teamId")
        if self._default_id is not None:
            info = self._api.team.get_info_by_id(self._default_id, raise_error=True)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {}
        res["options"] = {
            "showLabel": self._show_label,
            "filterable": True,
            "showWorkspace": False,
            "showTeam": True,
            "onlyAvailable": True,
        }
        if self._size is not None:
            res["options"]["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        return {
            "teamId": self._default_id,
        }

    def get_selected_id(self):
        return StateJson()[self.widget_id]["teamId"]
