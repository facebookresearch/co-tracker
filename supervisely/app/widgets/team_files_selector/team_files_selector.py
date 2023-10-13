from typing import Dict, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api


class TeamFilesSelector(Widget):
    def __init__(
        self,
        team_id: int,
        multiple_selection: bool = False,
        max_height: int = 500,
        selection_file_type: Literal["folder", "file"] = None,
        hide_header: bool = True,
        hide_empty_table: bool = True,
        additional_fields: List[
            Literal["id", "createdAt", "updatedAt", "type", "size", "mimeType"]
        ] = [],
        widget_id: str = None,
    ):
        self._api = Api()
        self._team_id = team_id

        self._multiple_selection = multiple_selection
        self._max_height = f"{max_height}px"
        self._selection_file_type = selection_file_type
        self._hide_header = hide_header
        self._hide_empty_table = hide_empty_table

        available_fields = ["id", "createdAt", "updatedAt", "type", "size", "mimeType"]
        for field in additional_fields:
            if field not in available_fields:
                raise ValueError(
                    f'"{field}" is not a valid field. Available fields: {available_fields}.'
                )

        self._additional_fields = additional_fields
        self._selected = []

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "teamId": self._team_id,
            "options": {
                "multipleSelection": self._multiple_selection,
                "maxHeight": self._max_height,
                "selectionFileType": self._selection_file_type,
                "hideHeader": self._hide_header,
                "hideEmptyTable": self._hide_empty_table,
                "additionalFields": self._additional_fields,
            },
        }

    def get_json_state(self) -> Dict:
        return {"selected": self._selected}

    def get_selected_items(self) -> List[dict]:
        selected = StateJson()[self.widget_id]["selected"]
        return [item for item in selected]

    def get_selected_paths(self) -> List[str]:
        selected = StateJson()[self.widget_id]["selected"]
        return [item["path"] for item in selected]

    def set_team_id(self, team_id: int):
        if type(team_id) != int:
            raise ValueError(f"team_id must be int, got {type(team_id)}")
        if team_id == self._team_id:
            return
        DataJson()[self.widget_id]["teamId"] = None
        DataJson().send_changes()
        self._team_id = team_id
        DataJson()[self.widget_id]["teamId"] = self._team_id
        DataJson().send_changes()
