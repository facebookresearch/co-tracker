from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.app.widgets_context import JinjaWidgets


class VideoPlayer(Widget):
    def __init__(self, url: str = None, mime_type: str = "video/mp4", widget_id: str = None):

        self._api = Api()

        self._url = url
        self._mime_type = mime_type
        self._mask_path = None

        self._current_timestamp = 0
        self._is_playing = False

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/video_player/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def get_json_data(self):
        return {
            "url": self._url,
            "mimeType": self._mime_type,
            "maskPath": self._mask_path,
        }

    def get_json_state(self):
        return {
            "currentTime": 0,
            "timeToSet": 0,
            "isPlaying": False,
        }

    @property
    def url(self):
        return self._url

    @property
    def mime_type(self):
        return self._mime_type

    def set_video(self, url: str, mime_type: str = "video/mp4"):
        self._url = url
        self._mime_type = mime_type
        DataJson()[self.widget_id]["url"] = self._url
        DataJson()[self.widget_id]["mimeType"] = self._mime_type
        DataJson().send_changes()
        StateJson()[self.widget_id]["currentTime"] = 0
        StateJson().send_changes()

    def play(self):
        is_playing = StateJson()[self.widget_id]["isPlaying"]
        if is_playing is True:
            return
        self._is_playing = True
        StateJson()[self.widget_id]["isPlaying"] = True
        StateJson().send_changes()

    def pause(self):
        is_playing = StateJson()[self.widget_id]["isPlaying"]
        if is_playing is False:
            return
        self._is_playing = False
        StateJson()[self.widget_id]["isPlaying"] = False
        StateJson().send_changes()

    def get_current_timestamp(self):
        self._current_timestamp = round(StateJson()[self.widget_id]["currentTime"], 1)
        return self._current_timestamp

    def set_current_timestamp(self, value: int):
        self._current_timestamp = value
        StateJson()[self.widget_id]["timeToSet"] = value
        StateJson().send_changes()
        return self._current_timestamp

    def draw_mask(self, path):
        self._mask_path = path
        DataJson()[self.widget_id]["maskPath"] = self._mask_path
        DataJson().send_changes()

    def hide_mask(self):
        self._mask_path = None
        DataJson()[self.widget_id]["maskPath"] = self._mask_path
        DataJson().send_changes()
