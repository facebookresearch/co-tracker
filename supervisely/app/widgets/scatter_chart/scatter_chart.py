from typing import Union
from functools import wraps
from supervisely.app.widgets.apexchart.apexchart import Apexchart
from supervisely.app.content import StateJson, DataJson

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

"""
size1 = 10
xy = np.random.normal(15, 6, (size1, 2)).tolist()
s1 = [{"x": x, "y": y} for x, y in xy]

size2 = 30
x2 = list(range(size2))
y2 = np.random.uniform(low=0, high=30, size=size2).tolist()
s2 = [{"x": x, "y": y} for x, y in zip(x2, y2)]

scatter_chart = ScatterChart(
    title="Max vs Denis",
    series=[{"name": "Max", "data": s1}, {"name": "Denis", "data": s2}],
    xaxis_type="numeric",
)

@scatter_chart.click
def on_click(datapoint: ScatterChart.ClickedDataPoint):
    print(f"Line: {datapoint.series_name}")
    print(f"x = {datapoint.x}")
    print(f"y = {datapoint.y}")
"""


class ScatterChart(Apexchart):
    def __init__(
        self,
        title: str,
        series: list = [],
        zoom: bool = True,
        markers_size: int = 4,
        data_labels: bool = False,
        xaxis_type: Literal["numeric", "category", "datetime"] = "numeric",
        xaxis_title: str = None,
        yaxis_title: str = None,
        yaxis_autorescale: bool = True,  # issue in apex, need to refresh page
        height: Union[int, str] = 350,
        decimalsInFloat: int = 2,
    ):
        self._title = title
        self._series = series
        self._zoom = zoom
        self._markers_size = markers_size
        self._data_labels = data_labels
        self._xaxis_type = xaxis_type
        self._xaxis_title = xaxis_title
        self._yaxis_title = yaxis_title
        self._yaxis_autorescale = yaxis_autorescale
        self._ymin = 0
        self._ymax = 10
        self._widget_height = height
        self._decimalsInFloat = decimalsInFloat

        self._options = {
            "chart": {
                "type": "scatter",
                "zoom": {"enabled": self._zoom, "type": "xy"},
                "animations": {
                    # TODO: fix animations bug in scatter_chart (issue with DataJson)
                    "enabled": False,
                },
            },
            "dataLabels": {"enabled": self._data_labels},
            "stroke": {"width": 0},
            "title": {"text": self._title, "align": "left"},
            "grid": {"row": {"colors": ["#f3f3f3", "transparent"], "opacity": 0.5}},
            "xaxis": {"type": self._xaxis_type},
            "markers": {"size": self._markers_size},
            "yaxis": [{"show": True, "decimalsInFloat": self._decimalsInFloat}],
        }
        if self._xaxis_title is not None:
            self._options["xaxis"]["title"] = {"text": str(self._xaxis_title)}
        if self._yaxis_title is not None:
            self._options["yaxis"][0]["title"] = {"text": self._yaxis_title}

        super().__init__(
            series=self._series,
            options=self._options,
            type="scatter",
            height=self._widget_height,
        )
        self.update_y_range(self._ymin, self._ymax)

    def update_y_range(self, ymin: int, ymax: int, send_changes=True):
        self._ymin = min(self._ymin, ymin)
        self._ymax = max(self._ymax, ymax)
        if self._yaxis_autorescale is False:
            self._options["yaxis"][0]["min"] = self._ymin
            self._options["yaxis"][0]["max"] = self._ymax

            self.update_data()
            if send_changes is True:
                DataJson().send_changes()

    def add_series(self, name: str, x: list, y: list):
        # print(self._options["yaxis"]["min"], self._options["yaxis"]["max"])
        super().add_series(name, x, y)
        self.update_y_range(min(y), max(y))

    def add_series_batch(self, series: dict):
        # usage example
        # lines = []
        # for class_name, x, y in stats.get_series():
        #     lines.append({"name": class_name, "x": x, "y": y})
        for serie in series:
            name = serie["name"]
            x = serie["x"]
            y = serie["y"]
            super().add_series(name, x, y, send_changes=False)
            self.update_y_range(min(y), max(y), send_changes=False)
        DataJson().send_changes()

    def add_points_to_serie(
        self, name: str, x: Union[list, int], y: Union[list, int], send_changes=True
    ):
        if isinstance(x, int):
            x = [x]
            y = [y]
        data = [{"x": px, "y": py} for px, py in zip(x, y)]
        for serie in self._series:
            if serie["name"] == name:
                serie["data"] += data
                break
        self.update_data()
        if send_changes:
            DataJson().send_changes()
