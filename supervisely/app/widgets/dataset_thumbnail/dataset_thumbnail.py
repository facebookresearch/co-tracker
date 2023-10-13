from supervisely.app import DataJson
from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import Widget
from supervisely.api.project_api import ProjectInfo
from supervisely.api.dataset_api import DatasetInfo
from supervisely.project.project import Project, Dataset


class DatasetThumbnail(Widget):
    def __init__(
        self,
        project_info: ProjectInfo = None,
        dataset_info: DatasetInfo = None,
        show_project_name: bool = True,
        widget_id: str = None,
    ):
        self._project_info: ProjectInfo = None
        self._dataset_info: DatasetInfo = None
        self._id: int = None
        self._name: str = None
        self._description: str = None
        self._url: str = None
        self._image_preview_url: str = None
        self._show_project_name: bool = show_project_name
        self._project_name: str = None
        self._project_url: str = None
        self._set_info(project_info, dataset_info, show_project_name)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "url": self._url,
            "image_preview_url": self._image_preview_url,
            "show_project_name": self._show_project_name,
            "project_name": self._project_name,
            "project_url": self._project_url,
        }

    def get_json_state(self):
        return None

    def _set_info(
        self, project_info: ProjectInfo, dataset_info: DatasetInfo, show_project_name: bool
    ):
        if project_info is None:
            return
        if dataset_info is None:
            return

        self._project_info = project_info
        self._dataset_info = dataset_info
        self._id = dataset_info.id
        self._name = dataset_info.name
        self._description = f"{self._dataset_info.items_count} {self._project_info.type} in dataset"
        self._url = Dataset.get_url(project_id=project_info.id, dataset_id=dataset_info.id)
        self._image_preview_url = dataset_info.image_preview_url
        self._show_project_name = show_project_name
        self._project_name = project_info.name
        self._project_url = Project.get_url(project_info.id)

    def set(
        self, project_info: ProjectInfo, dataset_info: DatasetInfo, show_project_name: bool = True
    ):
        self._set_info(project_info, dataset_info, show_project_name)
        self.update_data()
        DataJson().send_changes()
