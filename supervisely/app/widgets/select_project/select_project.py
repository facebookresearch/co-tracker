from typing import Dict, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app.widgets import (
    Widget,
    SelectWorkspace,
    generate_id,
)
from supervisely.app import DataJson, StateJson
from supervisely.api.api import Api
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.app.widgets.select_sly_utils import _get_int_or_env


class SelectProject(Widget):
    def __init__(
        self,
        default_id: int = None,
        workspace_id: int = None,
        compact: bool = False,
        allowed_types: List[ProjectType] = [],
        show_label: bool = True,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
    ):
        self._api = Api()
        self._default_id = default_id
        self._ws_id = workspace_id
        self._team_id = None
        self._compact = compact
        self._project_types = allowed_types
        self._show_label = show_label
        self._size = size
        self._ws_selector = None
        self._disabled = False

        self._default_id = _get_int_or_env(self._default_id, "modal.state.slyProjectId")
        if self._default_id is not None:
            info = self._api.project.get_info_by_id(self._default_id, raise_error=True)
            self._ws_id = info.workspace_id
        self._ws_id = _get_int_or_env(self._ws_id, "context.workspaceId")

        if self._ws_id is not None:
            info = self._api.workspace.get_info_by_id(self._ws_id)
            self._team_id = info.team_id
        self._team_id = _get_int_or_env(self._team_id, "context.teamId")

        if compact is True:
            if self._ws_id is None:
                raise ValueError(
                    '"workspace_id" have to be passed as argument or "compact" has to be False'
                )
        else:
            # if self._show_label is False:
            #     logger.warn(
            #         "show_label can not be false if compact is True and default_id / workspace_id are not defined"
            #     )
            self._show_label = True
            self._ws_selector = SelectWorkspace(
                default_id=self._ws_id,
                team_id=self._team_id,
                compact=False,
                show_label=True,
                size=self._size,
                widget_id=generate_id(),
            )
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {}
        res["disabled"] = self._disabled
        res["workspaceId"] = self._ws_id
        res["options"] = {
            "showLabel": self._show_label,
            "compact": self._compact,
            "filterable": True,
            "availableTypes": [ptype.value for ptype in self._project_types],
        }
        if self._size is not None:
            res["options"]["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        return {
            "projectId": self._default_id,
        }

    def get_selected_id(self):
        return StateJson()[self.widget_id]["projectId"]

    def disable(self):
        if self._compact is False:
            self._ws_selector.disable()
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        if self._compact is False:
            self._ws_selector.enable()
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()
