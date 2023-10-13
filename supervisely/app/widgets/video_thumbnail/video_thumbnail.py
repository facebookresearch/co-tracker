from supervisely.app.widgets import Widget
from supervisely.app import DataJson
from supervisely.api.api import Api
from supervisely.api.video.video_api import VideoInfo
from supervisely.project.project import Project
from supervisely.video.video import get_labeling_tool_url, get_labeling_tool_link


class VideoThumbnail(Widget):
    def __init__(self, info: VideoInfo = None, widget_id: str = None):
        self._info: VideoInfo = None
        self._id: int = None
        self._name: str = None
        self._description: str = None
        self._url: str = None
        self._open_link: str = None
        self._preview_url: str = None
        self._set_info(info)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "url": self._url,
            "link": self._open_link,
            "image_preview_url": self._preview_url,
        }

    def _set_info(self, info: VideoInfo = None):
        if info is None:
            return
        self._info = info
        self._id = info.id
        self._name = info.name
        self._description = (
            f"Video length: {info.duration_hms} / {info.frames_count_compact} frames"
        )
        self._url = get_labeling_tool_url(info.dataset_id, info.id)
        self._open_link = get_labeling_tool_link(self._url, "open video")
        self._preview_url = info.image_preview_url

    def set_video(self, info: VideoInfo):
        self._set_info(info)
        self.update_data()
        DataJson().send_changes()

    def set_video_id(self, id: int):
        api = Api()
        info = api.video.get_info_by_id(id, raise_error=True)
        self.set_video(info)

    def get_json_state(self):
        return None
