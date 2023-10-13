from typing import List, Optional, Union

from supervisely.api.api import Api
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class FileStorageUpload(Widget):
    def __init__(
        self,
        team_id: int,
        path: str,
        change_name_if_conflict: Optional[bool] = False,
        widget_id: str = None,
    ):
        self._api = Api()
        self._team_id = team_id
        self._change_name_if_conflict = change_name_if_conflict
        self._path = self._get_path(path)
        self._files = []

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_path(self, path: str):
        self._path = self._get_path(path)
        DataJson()[self.widget_id]["path"] = self._path
        DataJson().send_changes()

    def _get_path(self, path: str):
        if self._change_name_if_conflict is True:
            path = f"/{path}" if not path.startswith("/") else path
            return self._api.file.get_free_dir_name(self._team_id, path)
        return path

    def get_json_data(self):
        return {"team_id": self._team_id, "path": self._path}

    def get_json_state(self):
        return {"files": self._files}

    @property
    def path(self):
        return DataJson()[self.widget_id]["path"]

    @path.setter
    def path(self, path: str):
        self._set_path(path)

    def set_path(self, path: str):
        self._set_path(path)

    def get_team_id(self):
        return self._team_id

    def get_uploaded_paths(self) -> Union[List[str], None]:
        response = StateJson()[self.widget_id]["files"]
        if len(response) == 0 or response is None:
            return []
        uploaded_files = response["uploadedFiles"]
        if len(uploaded_files) == 0:
            return None
        paths = [file["path"] for file in uploaded_files]
        return paths

    # def get_uploaded_info(self) -> Union[List[sly.api.file_api.FileInfo], None]:
    #     response = StateJson()[self.widget_id]["files"]
    #     uploaded_files = response["uploadedFiles"]
    #     if len(uploaded_files) == 0:
    #         return None

    #     files_infos = []
    #     for item in uploaded_files:
    #         TODO: convert from json instead of api queries
    #         files_infos.append(self._api.file.get_info_by_id(item["id"]))
    #     return files_infos
