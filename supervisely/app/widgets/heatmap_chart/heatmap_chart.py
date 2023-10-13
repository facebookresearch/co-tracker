from typing import Union
from functools import wraps
from supervisely.app.widgets.apexchart.apexchart import Apexchart
from supervisely.app.content import StateJson, DataJson

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

"""
chart = sly.app.widgets.HeatmapChart(
    title="Objects count distribution for every class",
    xaxis_title="Number of objects on image",
    color_range="row",
    tooltip="There are {y} images with {x} objects of class {series_name}",
)
"""


class HeatmapChart(Apexchart):
    def __init__(
        self,
        title: str,
        data_labels: bool = True,
        xaxis_title: str = None,
        color_range: Literal["table", "row"] = "row",
        tooltip: str = None,
    ):
        self._title = title
        self._series = []
        self._original_series_x = {}
        self._data_labels = data_labels
        self._xaxis_title = xaxis_title
        self._widget_height = 350
        self._color_range = color_range
        self._distributed = False
        self._colors = ["#008FFB"]
        self._tooltip = tooltip
        if self._color_range == "row":
            self._distributed = True
            self._colors = [
                "#F3B415",
                "#F27036",
                "#663F59",
                "#6A6E94",
                "#4E88B4",
                "#00A7C6",
                "#18D8D8",
                "#A9D794",
                "#46AF78",
                "#A93F55",
                "#8C5E58",
                "#2176FF",
                "#33A1FD",
                "#7A918D",
                # "#BAFF29",
                "#bce95c",
            ]

        self._options = {
            "chart": {
                "type": "heatmap",
                "toolbar": {
                    "tools": {
                        "download": True,
                        "selection": False,
                        "zoom": False,
                        "zoomin": False,
                        "zoomout": False,
                        "pan": False,
                        "reset": False,
                    },
                },
            },
            "legend": {"show": False},
            "plotOptions": {
                "heatmap": {
                    "distributed": self._distributed,
                    "colorScale": {
                        "ranges": [
                            {"from": 0, "to": 0, "name": "", "color": "#DCDCDC"},
                        ]
                    },
                },
            },
            "dataLabels": {"enabled": self._data_labels, "style": {"colors": ["#fff"]}},
            "colors": self._colors,
            "title": {"text": self._title, "align": "left"},
            "xaxis": {},
        }
        if self._xaxis_title is not None:
            self._options["xaxis"]["title"] = {"text": str(self._xaxis_title)}

        sly_options = {}
        if self._tooltip is not None:
            sly_options["tooltip"] = self._tooltip
            # self._options["tooltip"] = {"followCursor": False}
            self._options["tooltip"] = {"y": {}}

        super(HeatmapChart, self).__init__(
            series=self._series,
            options=self._options,
            type="heatmap",
            height=self._widget_height,
            sly_options=sly_options,
        )

    def _update_height(self):
        self._widget_height = len(self._series) * 40 + 70
        DataJson()[self.widget_id]["height"] = self._widget_height

    def add_series_batch(self, series: dict):
        # usage example
        # lines = []
        # for class_name, x, y in stats.get_series():
        #     lines.append({"name": class_name, "x": x, "y": y})
        for serie in series:
            name = serie["name"]
            x = serie["x"]
            y = serie["y"]
            self.add_series(name, x, y, send_changes=False)
        DataJson().send_changes()

    def add_series(self, name: str, x: list, y: list, send_changes=True):
        x_str = [str(num) for num in x]
        self._original_series_x[name] = x.copy()
        super().add_series(name, x_str, y, send_changes)
        self._update_height()

    def get_clicked_datapoint(self):
        res = super().get_clicked_datapoint()
        if res is not None:
            res = res._replace(x=self._original_series_x[res.series_name][res.data_index])
        return res
