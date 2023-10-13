from typing import Dict, Union
from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely import is_development


class TaskLogs(Widget):
    def __init__(
        self,
        task_id: int = None,
        widget_id: str = None,
    ):
        self._task_id = task_id
        self._is_development = is_development()
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"taskId": self._task_id}

    def get_json_state(self) -> Dict:
        return {}

    def get_task_id(self) -> int:
        return DataJson()[self.widget_id]["taskId"]

    def _set_task_id(self, task_id: Union[int, None]):
        self._task_id = task_id
        DataJson()[self.widget_id]["taskId"] = self._task_id
        DataJson().send_changes()

    def set_task_id(self, task_id: int):
        self._set_task_id(None)
        if type(task_id) != int:
            raise TypeError(f"task_id must be int, but {type(task_id)} was given")
        self._set_task_id(task_id)
