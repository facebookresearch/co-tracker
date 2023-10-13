# <div v-if="data.progress.message" class="mt10">
#     <div style="color: #20a0ff">
#         {{data.progress.message}}: {{data.progress.current}} / {{data.progress.total}}
#     </div>
#     <el-progress :percentage="data.progress.percent"></el-progress>
# </div>

import math
from supervisely.api.api import Api
from supervisely.task.progress import Progress


class ProgressBar:
    def __init__(self, task_id, api: Api, v_model, message, total=None, is_size=False, min_report_percent=1):
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        if self._v_model.startswith("data.") is False:
            new_v_model = v_model.replace('state.', 'data.', 1)
            raise KeyError(f"Data for this widget has to be stored in data field. "
                           f"Please, change value of v_model argument "
                           f"from {v_model} to {new_v_model} manually and also check html template")

        self._message = message
        self._is_size = is_size
        self._min_report_percent = min_report_percent
        self._total = None
        self._progress: Progress = None
        self.set_total(total)

    def get_total(self):
        return self._total

    def set_total(self, total):
        self._total = total
        if total is not None:
            self._progress = Progress(
                self._message,
                self._total,
                is_size=self._is_size,
                min_report_percent=self._min_report_percent
            )
        else:
            self._progress = None

    def init_data(self, data: dict):
        data[self._v_model.split(".")[1]] = self.to_json()

    def to_json(self):
        res = {
            "message": self._progress.message if self._progress is not None else None,
            "current": self._progress.current_label if self._progress is not None else None,
            "total": self._progress.total_label if self._progress is not None else None,
            "percent": math.floor(self._progress.current * 100 / self._progress.total) if self._progress is not None else None,
        }
        return res

    def get_field(self):
        return {"field": self._v_model, "payload": self.to_json()}

    def reset(self):
        self._progress = None

    def reset_and_update(self):
        self.reset()
        self.update(force=True)

    def is_reset(self):
        return self._progress is None

    def update(self, force=False):
        if self._progress is not None and (self._progress.need_report() or force):
            self._progress.report_progress()
            self._api.app.set_fields(self._task_id, [self.get_field()])
        if self._progress is None:
            self._api.app.set_fields(self._task_id, [self.get_field()])

    def set(self, value, force_update=False):
        if self._progress is None:
            self.set_total(self._total)
        self._progress.set_current_value(value)
        self.update(force_update)

    def increment(self, count, force_update=False):
        if self._progress is None:
            self.set_total(self._total)
        self._progress.iters_done(count)
        self.update(force_update)
