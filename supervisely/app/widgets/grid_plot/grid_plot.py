from typing import List, Dict
from supervisely.app.widgets import Widget, Container, generate_id, LinePlot
from supervisely.sly_logger import logger
from supervisely._utils import batched, rand_str
from supervisely.app.widgets.empty.empty import Empty
from supervisely.app.content import StateJson, DataJson


class GridPlot(Widget):
    def __init__(
        self,
        data: List[Dict or str] = [],
        columns: int = 1,
        gap: int = 10,
        widget_id: str = None,
    ):
        self._widgets = {}
        self._columns = columns
        self._gap = gap

        for plot_data in data:
            if isinstance(plot_data, dict):
                # self._widgets[plot_data['title']] = LinePlot(title=plot_data['title'], series=plot_data.get('series', []), show_legend=plot_data.get('show_legend', True))
                # передача параметров таким образом в конечном итоге приводит к ошибке JsonPatchConflict
                self._widgets[plot_data["title"]] = LinePlot(**plot_data)
            else:
                self._widgets[plot_data] = LinePlot(title=plot_data, series=[])

        if self._columns < 1:
            raise ValueError(f"columns ({self._columns}) < 1")
        if self._columns > len(self._widgets):
            logger.warn(
                f"Number of columns ({self._columns}) > number of widgets ({len(self._widgets)}). Columns are set to {len(self._widgets)}"
            )
            self._columns = len(self._widgets)

        self._content = None
        if self._columns == 1:
            self._content = Container(
                direction="vertical",
                widgets=self._widgets.values(),
                gap=self._gap,
                widget_id=generate_id(),
            )
        else:
            rows = []
            num_empty = len(self._widgets) % self._columns
            for i in range(num_empty):
                self._widgets[generate_id()] = Empty()
            for batch in batched(list(self._widgets.values()), batch_size=self._columns):
                rows.append(
                    Container(
                        direction="horizontal",
                        widgets=batch,
                        gap=self._gap,
                        fractions=[1] * len(batch),
                        widget_id=generate_id(),
                        overflow=None,
                    )
                )
            self._content = Container(
                direction="vertical",
                widgets=rows,
                gap=self._gap,
                widget_id=generate_id(),
            )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None

    def add_scalar(self, identifier: str, y, x):
        """
        Add scalar to series on plot. If no series with name,
         defined in `identifier` exists,
         one will be created automatically.

        :param identifier: slash-separated plot and series name
        :type identifier: str
        :param y: y value
        :type y: float | int
        :param x: x value
        :type x: float | int
        """
        plot_title, series_name = identifier.split("/")
        _, series = self._widgets[plot_title].get_series_by_name(series_name)

        if series is not None:
            self._widgets[plot_title].add_to_series(name_or_id=series_name, data=[(x, y)])
        else:
            self._widgets[plot_title].add_series(name=series_name, x=[x], y=[y])

    def add_scalars(self, plot_title: str, new_values: dict, x):
        """
        Add scalars to several series on one plot at point `x`

        :param plot_title: name of existing plot
        :type plot_title: str
        :param new_values: dictionary in the `{series_name: y_value, ...}` format
        :type new_values: dict
        :param x: value of `x`
        :type x: float | int
        """
        for series_name in new_values.keys():
            _, series = self._widgets[plot_title].get_series_by_name(series_name)
            if series is not None:
                self._widgets[plot_title].add_to_series(
                    name_or_id=series_name,
                    # data=[{"x": x, "y": new_values[series_name]}],
                    data=[(x, new_values[series_name])],
                )
            else:
                self._widgets[plot_title].add_series(
                    name=series_name, x=[x], y=[new_values[series_name]]
                )
