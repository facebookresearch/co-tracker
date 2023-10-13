from typing import List, Dict, Union
from supervisely.app.jinja2 import create_env
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class RadioTable(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        columns: List[str],
        rows: List[List[str]],
        subtitles: Union[Dict[str, str], List] = {},  # col_name -> subtitle
        column_formatters: Dict = {},
        widget_id: str = None,
    ):

        self._columns = columns
        self._rows = rows
        if len(subtitles) > 0:
            if isinstance(subtitles, dict):
                subtitles = [subtitles[col] for col in columns]
        else:
            subtitles = [None] * len(columns)
        self._subtitles = subtitles
        self._column_formatters = column_formatters

        self._header = []

        for col, subtitle in zip(columns, subtitles):
            self._header.append({"title": col, "subtitle": subtitle})

        self._frows = []

        super().__init__(widget_id=widget_id, file_path=__file__)

        self.rows = rows
        self._changes_handled = False

    def get_json_data(self):
        return {
            "header": self._header,
            "frows": self._frows,
            "raw_rows_data": self.rows,
        }

    def get_json_state(self):
        return {"selectedRow": 0}

    def format_value(self, column_name: str, value):
        fn = self._column_formatters.get(column_name, self.default_formatter)
        return fn(f"data.{self.widget_id}.raw_rows_data[params.ridx][params.vidx]")

    def default_formatter(self, value):
        if value is None:
            return "-"
        return "<div> {{{{ data.{}.raw_rows_data[params.ridx][params.vidx] }}}} </div>".format(
            self.widget_id
        )

    def _update_frows(self):
        self._frows = []
        for idx, row in enumerate(self._rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"Row #{idx} length is {len(row)} != number of columns ({len(self.columns)})"
                )
            frow = []
            for col, val in zip(self.columns, row):
                frow.append(self.format_value(col, val))
            self._frows.append(frow)

    def get_selected_row(self, state=StateJson()):
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            selected_row_index = widget_actual_state["selectedRow"]
            return self.rows[selected_row_index]

    def get_selected_row_index(self, state=StateJson()):
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            return widget_actual_state["selectedRow"]

    def value_changed(self, func):
        route_path = self.get_route_path(RadioTable.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True
        print(self._changes_handled)

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_row()
            func(res)

        return _value_changed

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def subtitles(self) -> List[str]:
        return self._subtitles

    def set_columns(
        self, columns: List[str], subtitles: Union[Dict[str, str], List[str]] = {}
    ) -> None:
        self._columns = columns
        if isinstance(subtitles, dict):
            subtitles = [subtitles[col] for col in columns]
        self._subtitles = subtitles
        self._header = []
        for col, subtitle in zip(columns, subtitles):
            self._header.append({"title": col, "subtitle": subtitle})
        DataJson()[self.widget_id]["header"] = self._header
        DataJson().send_changes()
        self.rows = [[] * len(columns)]

    def set_data(
        self,
        columns: List[str],
        rows: List[List[str]],
        subtitles: Union[Dict[str, str], List[str]] = {},
    ) -> None:
        self._columns = columns
        if isinstance(subtitles, dict):
            subtitles = [subtitles[col] for col in columns]
        self._subtitles = subtitles
        self._header = []
        for col, subtitle in zip(columns, subtitles):
            self._header.append({"title": col, "subtitle": subtitle})
        DataJson()[self.widget_id]["header"] = self._header
        DataJson().send_changes()
        self.rows = rows

    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, value):
        self._rows = value
        self._update_frows()
        DataJson()[self.widget_id]["frows"] = self._frows
        DataJson()[self.widget_id]["raw_rows_data"] = self._rows
        DataJson().send_changes()

    def select_row(self, row_index):
        if row_index < 0 or row_index > len(self._rows) - 1:
            raise ValueError(f'Row with index "{row_index}" does not exist')
        StateJson()[self.widget_id]["selectedRow"] = row_index
        StateJson().send_changes()
