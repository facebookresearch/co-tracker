from supervisely.api.api import Api


class Chart:
    def __init__(self, task_id, api: Api, v_model, title, series_names=None,
                 smoothing=None, yrange=None, ydecimals=None, xdecimals=None):
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        if self._v_model.startswith("data.") is False:
            new_v_model = v_model.replace('state.', 'data.', 1)
            raise KeyError(f"Data for this widget has to be stored in data field. "
                           f"Please, change value of v_model argument "
                           f"from {v_model} to {new_v_model} manually and also check html template")

        self._title = title
        if series_names is None:
            self._series_names = [title]
        else:
            self._series_names = series_names
        self._smoothing = smoothing
        self._yrange = yrange
        self._ydecimals = ydecimals
        self._xdecimals = xdecimals

        self._series = []
        for name in self._series_names:
            self._series.append({
                "name": name,
                "data": []
            })

    def to_json(self):
        result = {
            "options": {
                "title": self._title,
                # "groupKey": "my-synced-charts",
            },
            "series": self._series
        }
        if self._smoothing is not None:
            result["options"]["smoothingWeight"] = self._smoothing
        if self._yrange is not None:
            result["options"]["yaxisInterval"] = self._yrange
        if self._ydecimals is not None:
            result["options"]["decimalsInFloat"] = self._ydecimals
        if self._xdecimals is not None:
            result["options"]["xaxisDecimalsInFloat"] = self._xdecimals
        return result

    def init_data(self, data: dict):
        data[self._v_model.split(".")[1]] = self.to_json()

    def get_field(self, x, y, series_name=None):
        if series_name is None:
            series_name = self._title
        try:
            index = self._series_names.index(series_name)
        except ValueError:
            raise ValueError(f"Series '{series_name}' not found in {self._series_names}")
        return {"field": f"{self._v_model}.series[{index}].data", "payload": [[x, y]], "append": True}

    def append(self, x, y, series_name=None):
        field = self.get_field(x, y, series_name)
        self._api.app.set_fields(self._task_id, [field])