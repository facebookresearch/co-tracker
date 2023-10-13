try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.project.project_type import ProjectType


class DestinationProject(Widget):

    def __init__(
        self,
        workspace_id: int,
        project_type: Literal[
            ProjectType.IMAGES,
            ProjectType.VIDEOS,
            ProjectType.VOLUMES,
            ProjectType.POINT_CLOUDS,
            ProjectType.POINT_CLOUD_EPISODES,
        ] = ProjectType.IMAGES,
        widget_id: str = None,
    ):
        self._api = Api()

        self._project_mode = "new_project"
        self._dataset_mode = "new_dataset"

        self._project_id = None
        self._dataset_id = None

        self._project_name = ""
        self._dataset_name = ""
        
        self._use_project_datasets_structure = False
        
        self._workspace_id = workspace_id
        self._project_type = str(project_type)
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return {
            "workspace_id": str(self._workspace_id),
            "project_mode": self._project_mode,
            "project_id": self._project_id,
            "project_name": self._project_name,
            "project_type": self._project_type,
            "dataset_mode": self._dataset_mode,
            "dataset_id": self._dataset_id,
            "dataset_name": self._dataset_name,
            "use_project_datasets_structure": self._use_project_datasets_structure,
        }

    def get_selected_project_id(self):
        return StateJson()[self.widget_id]["project_id"]

    def get_selected_dataset_id(self):
        project_id = StateJson()[self.widget_id]["project_id"]
        dataset_mode = StateJson()[self.widget_id]["dataset_mode"]
        ds_name = StateJson()[self.widget_id]["dataset_id"]
        if project_id is not None and dataset_mode == "existing_dataset" and ds_name is not None:
            ds = self._api.dataset.get_info_by_name(parent_id=project_id, name=ds_name)
            return ds.id
        return None

    def get_project_name(self):
        return StateJson()[self.widget_id]["project_name"]

    def get_dataset_name(self):
        return StateJson()[self.widget_id]["dataset_name"]

    def use_project_datasets_structure(self):
        return StateJson()[self.widget_id]["use_project_datasets_structure"]
