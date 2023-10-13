from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson, DataJson
from typing import List, Tuple, Union

NumT = Union[int, float]


class LinePlot(Widget):
    def __init__(
        self,
        title: str,
        series: List[dict] = [],
        smoothing_weight: float = 0,
        group_key: str = None,
        show_legend: bool = True,
        decimals_in_float: int = 2,
        xaxis_decimals_in_float: int = None,
        yaxis_interval: list = None,
        widget_id=None,
        yaxis_autorescale: bool = True,  # issue in apex, need to refresh page
    ):
        """
        Create line plot.

        :param title: plot title
        :type title: str
        :param series: List of dicts in format
            `[{'name': series_name, 'data': [(x1, y1), ...]}]` or
            `[{'name': series_name, 'data': [{'x': x1, 'y': y1}, ...]}]`
        :type series: List[dict]
        :param smoothing_weight: smoothing coeficient; float number from [0, 1]
        :type smoothing_weight: float
        :param group_key: _description_
        :type group_key: str
        :param show_legend: _description_
        :type show_legend: bool
        :param decimals_in_float: number of decimal places for y-axis
        :type decimals_in_float: int
        :param xaxis_decimals_in_float: number of decimal places for x-axis
        :type xaxis_decimals_in_float: int
        :param yaxis_interval: _description_
        :type yaxis_interval: list
        :param widget_id: _description_
        :type widget_id: _type_
        :param yaxis_autorescale: _description_
        :type yaxis_autorescale: bool
        """
        self._title = title
        self._series = self._check_series(series)
        self._smoothing_weight = smoothing_weight
        self._group_key = group_key
        self._show_legend = show_legend
        self._decimals_in_float = decimals_in_float
        self._xaxis_decimals_in_float = xaxis_decimals_in_float
        self._yaxis_interval = yaxis_interval
        self._options = {
            "title": self._title,
            "smoothingWeight": self._smoothing_weight,
            "groupKey": self._group_key,
            "showLegend": self._show_legend,
            "decimalsInFloat": self._decimals_in_float,
            "xaxisDecimalsInFloat": self._xaxis_decimals_in_float,
            "yaxisInterval": self._yaxis_interval,
        }
        self._yaxis_autorescale = yaxis_autorescale
        self._ymin = 0
        self._ymax = 10
        super(LinePlot, self).__init__(widget_id=widget_id, file_path=__file__)
        self.update_y_range(self._ymin, self._ymax)

    def get_json_data(self):
        return {
            "title": self._title,
            "series": self._series,
            "options": self._options,
            "ymin": self._ymin,
            "ymax": self._ymax,
        }

    def get_json_state(self):
        return None

    def update_y_range(self, ymin: int, ymax: int, send_changes=True):
        self._ymin = min(self._ymin, ymin)
        self._ymax = max(self._ymax, ymax)
        if self._yaxis_autorescale is False:
            self._options["yaxis"][0]["min"] = self._ymin
            self._options["yaxis"][0]["max"] = self._ymax

    def add_series(self, name: str, x: list, y: list, send_changes: bool = True):
        assert len(x) == len(y), ValueError(
            f"Lists x and y have different lenght, {len(x)} != {len(y)}"
        )

        # data = [{"x": px, "y": py} for px, py in zip(x, y)]
        data = [(px, py) for px, py in zip(x, y)]
        series = {"name": name, "data": data}
        self._series.append(series)

        if len(y) > 0:
            self.update_y_range(min(y), max(y))

        DataJson()[self.widget_id]["series"] = self._series
        if send_changes:
            DataJson().send_changes()

    def add_series_batch(self, series: list):
        for serie in series:
            name = serie["name"]
            x = serie["x"]
            y = serie["y"]
            self.add_series(name, x, y, send_changes=False)
        DataJson().send_changes()

    def add_to_series(
        self,
        name_or_id: Union[str, int],
        data: Union[List[Union[tuple, dict]], Union[tuple, dict]],
    ):
        """
        Add new points to series

        :param name_or_id: series name
        :type name_or_id: str | int
        :param data: point or list of points to add; use one of the following formats
            `[(x1, y1), ...]`, `[{'x': x1, 'y': y1}, ...]`, `(x1,y1)` or `{'x': x1, 'y': y1}`
        :type data: Union[List[Union[tuple, dict]], Union[tuple, dict]]
        """
        if isinstance(name_or_id, int):
            series_id = name_or_id
        else:
            series_id, _ = self.get_series_by_name(name_or_id)

        if isinstance(data, List):
            data_list = self._list_of_point_dicts_to_list_of_tuples(data)
        else:
            # single datapoint
            data_list = self._list_of_point_dicts_to_list_of_tuples([data])

        self._series[series_id]["data"].extend(data_list)
        DataJson()[self.widget_id]["series"] = self._series
        DataJson().send_changes()

    def get_series_by_name(self, name):
        series_list = DataJson()[self.widget_id]["series"]
        series_id, series_data = next(
            ((i, series) for i, series in enumerate(series_list) if series["name"] == name),
            (None, None),
        )
        # assert series_id is not None, KeyError("Series with name: {name} doesn't exists.")
        return series_id, series_data

    def clean_up(self):
        self._series = []
        self._ymin = 0
        self._ymax = 10
        DataJson()[self.widget_id]["series"] = self._series
        DataJson()[self.widget_id]["ymin"] = self._ymin
        DataJson()[self.widget_id]["ymax"] = self._ymax
        DataJson().send_changes()

    def _point_dict_to_tuple(self, point_dct: dict) -> Tuple[NumT, NumT]:
        return (point_dct["x"], point_dct["y"])

    def _list_of_point_dicts_to_list_of_tuples(
        self, point_dcts: List[Union[dict, tuple]]
    ) -> List[Tuple[NumT, NumT]]:
        if len(point_dcts) == 0:
            return []
        if isinstance(point_dcts[0], tuple):
            return point_dcts
        return [self._point_dict_to_tuple(dct) for dct in point_dcts]

    def _check_series(self, series: List[dict]) -> List[dict]:
        for serie in series:
            serie["data"] = self._list_of_point_dicts_to_list_of_tuples(serie["data"])
        return series
