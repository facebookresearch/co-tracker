try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Any, Dict, Optional

import supervisely.io.env as sly_env
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.nn.inference import Session


class ModelInfo(Widget):
    def __init__(
        self,
        session_id: int = None,
        team_id: int = None,
        widget_id: str = None,
        replace_none_with: Optional[str] = None,
    ):
        self._api = Api()
        self._session_id = session_id
        self._team_id = team_id
        self._model_info = None
        # used only if session data recieved from Session instance directly
        self._replace_none_with = replace_none_with

        if self._team_id is None:
            self._team_id = sly_env.team_id()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        data = {}
        data["teamId"] = self._team_id
        if self._session_id is not None:
            data["model_connected"] = True
            if self._model_info is None:
                data["model_info"] = self._get_info()
            else:
                data["model_info"] = self._model_info
        elif self._session_id is None and self._model_info is not None:
            data["model_info"] = self._model_info
            data["model_connected"] = True
        else:
            data["model_info"] = None
            data["model_connected"] = False

        return data

    def get_json_state(self):
        state = {}
        state["sessionId"] = self._session_id
        return state

    def set_session_id(self, session_id: int):
        self._session_id = session_id
        self._model_info = self._get_info()
        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()

    def set_model_info(
        self,
        session_id: Optional[int] = None,
        model_info: Optional[dict] = None,
    ):
        if session_id is None and model_info is None:
            raise ValueError("Both session_id and model_info can't be None.")

        self._session_id = session_id
        self._model_info = model_info
        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()

    def _get_info(self):
        session = Session(self._api, self._session_id)
        return session.get_human_readable_info(self._replace_none_with)

    @property
    def session_id(self):
        return self._session_id
