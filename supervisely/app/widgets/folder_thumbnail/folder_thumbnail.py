import os
from supervisely.api.file_api import FileInfo
from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely._utils import abs_url


class FolderThumbnail(Widget):
    def __init__(self, info=None, widget_id: str = None):
        self._id: int = None
        self._info: info = None
        self._description: str = None
        self._url: str = None
        self._set_info(info)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "id": self._id,
            "name": "Folder in Team files",
            "description": self._description,
            "url": self._url,
            "description": self._description,
            "icon": {
                "className": "zmdi zmdi-folder-person",
                "color": "#4977ff",
                "bgColor": "#ddf2ff",
            },
        }

    def get_json_state(self):
        return None

    def _set_info(self, info: FileInfo = None):
        if info is None:
            return
        self._id = info.id
        self._info = info
        self._description = os.path.dirname(info.path)
        self._url = abs_url(f"/files/{info.id}")

    def set(self, info):
        self._set_info(info)
        self.update_data()
        DataJson().send_changes()
