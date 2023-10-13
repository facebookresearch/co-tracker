from typing import Union, NamedTuple, List, Dict
from supervisely.app.widgets.apexchart.apexchart import Apexchart
from supervisely.app.content import StateJson, DataJson

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

"""
series = [
    {"name": "Team A", "data": 44},
    {"name": "Team B", "data": 55},
    {"name": "Team C", "data": 13},
    {"name": "Team D", "data": 43},
    {"name": "Team E", "data": 22},
]

pie_chart = sly.app.widgets.PieChart(
    title="Simple Pie",
    series=series,
    stroke_width=1,
    data_labels=True,
    height=350,
    type="pie",
)

text = sly.app.widgets.Text("Click on the slice to see it's value", status="info")
container = sly.app.widgets.Container(widgets=[pie_chart, text])
card = sly.app.widgets.Card(title="Pie Chart", content=container)
app = sly.Application(layout=card)


@pie_chart.click
def show_selection(datapoint: sly.app.widgets.PieChart.ClickedDataPoint):
    data_name = datapoint.data["name"]
    data_value = datapoint.data["value"]
    data_index = datapoint.data_index
    text.set(f"Selected slice: {data_name}, Value: {data_value}, Index: {data_index}", "info")
"""


class PieChart(Apexchart):
    class ClickedDataPoint(NamedTuple):
        """Class, representing clicked datapoint, which contains information about series, data index and data itself.
        It will be returned after click event on datapoint in immutable namedtuple
        with fields: series_index, data_index, data."""

        series_index: int
        data_index: int
        data: dict

    def __init__(
        self,
        title: str,
        series: List[Dict[str, Union[int, float]]] = [],
        stroke_width: int = 2,
        data_labels: bool = False,
        height: Union[int, str] = 350,
        type: Literal["pie", "donut"] = "pie",
    ):
        self._title = title
        self._series = series
        self._stroke_width = stroke_width
        self._data_labels = data_labels
        self._widget_height = height

        self._series_labels = [serie["name"] for serie in self._series]
        self._series_data = [serie["data"] for serie in self._series]

        if type != "pie" and type != "donut":
            raise ValueError("type must be 'pie' or 'donut'")
        self._type = type

        self._options = {
            "labels": self._series_labels,
            "chart": {"type": self._type},
            "dataLabels": {"enabled": self._data_labels},
            "stroke": {"width": self._stroke_width},
            "title": {"text": self._title},
        }
        super(PieChart, self).__init__(
            series=self._series_data,
            options=self._options,
            type=self._type,
            height=self._widget_height,
        )

    def _manage_series(self, names: List[str], values: List[Union[int, float]], set: bool = False):
        """This is a private method, which should not be used directly. Use add_series() or set_series() instead.
        It will add or set series to the chart, depending on set parameter. If set is True, all previous series will be
        deleted, otherwise new series will be added to the chart.

        :param names: list of names of the slices in series, which will be displayed on the chart
        :type names: List[str]
        :param values: list of values of the slices in series, which will be displayed on the chart
        :type values: List[Union[int, float]
        :param set: if True, all previous series will be deleted, otherwise new series will be added to the chart,
            defaults to False
        :type set: bool, optional
        :raises ValueError: if names and values has different length
        :raises ValueError: if any of values is not int or float
        """
        if len(names) != len(values):
            raise ValueError(
                f"Names and values has different length: {len(names)} != {len(values)}"
            )
        for value in values:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"All values must be ints or floats, but {value} is {type(value)}."
                )

        if set:
            self._series = values
            self._options["labels"] = names
        else:
            self._series.extend(values)
            self._options["labels"].extend(names)

        self.update_data()
        DataJson().send_changes()

    def add_series(self, names: List[str], values: List[Union[int, float]]):
        # print(self._options["yaxis"]["min"], self._options["yaxis"]["max"])
        self._manage_series(names, values)

    def set_series(self, names: List[str], values: List[Union[int, float]]):
        """Sets series to the chart, deleting all previous series. Len of names and values must be equal,
        otherwise ValueError will be raised.

        :param names: list of names of the slices in series, which will be displayed on the chart
        :type names: List[str]
        :param values: list of values of the slices in series, which will be displayed on the chart
        :type values: List[Union[int, float]]
        """
        self._manage_series(names, values, set=True)

    def get_series(self, index: int) -> Dict[str, Union[str, int, float]]:
        """Returns series by index. If index is out of range, IndexError will be raised.
        Returned series is a dict with keys "name" and "data".

        :param index: index of the series, if index is out of range, IndexError will be raised
        :type index: int
        :raises TypeError: if index is not int
        :raises IndexError: if index is out of range
        :return: series name ans data by given index
        :rtype: Dict[str, Union[str, int, float]]
        """

        if not isinstance(index, int):
            raise TypeError(f"Index must be int, but {index} is {type(index)}.")
        try:
            return {"name": self._options["labels"][index], "data": self._series[index]}
        except IndexError:
            raise IndexError(f"Series with index {index} does not exist.")

    def delete_series(self, index: int):
        """Removes series by index from the chart. If index is out of range, IndexError will be raised.

        :param index: index of the series to delete
        :type index: int
        :raises TypeError: if index is not int
        :raises IndexError: if index is out of range
        """
        if not isinstance(index, int):
            raise TypeError(f"Index must be int, but {index} is {type(index)}.")
        try:
            del self._series[index]
            del self._options["labels"][index]
        except IndexError:
            raise IndexError(f"Series with index {index} does not exist.")
        self.update_data()
        DataJson().send_changes()

    def get_clicked_value(self):
        return StateJson()[self.widget_id]["clicked_value"]

    def get_clicked_datapoint(self) -> Union[ClickedDataPoint, None]:
        """Returns clicked datapoint as a ClickedDataPoint object, which is a namedtuple with fields:
        series_index, data_index and data. If click was outside of the slices, None will be returned.

        :return: clicked datapoint as a ClickedDataPoint object or None if click was outside of the slices
        :rtype: Union[ClickedDataPoint, None]
        """
        value = self.get_clicked_value()
        series_index = value["seriesIndex"]
        data_index = value["dataPointIndex"]

        if series_index == -1 or data_index == -1:
            # If click was outside of the slices.
            return

        data = {
            "name": self._options["labels"][data_index],
            "value": self._series[data_index],
        }

        res = self.ClickedDataPoint(series_index, data_index, data)
        return res
