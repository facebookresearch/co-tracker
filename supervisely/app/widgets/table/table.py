import copy
import io

import numpy as np
import pandas as pd
import re
from typing import NamedTuple, Any, List, Optional, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger
import traceback
from fastapi.responses import StreamingResponse, RedirectResponse


class PackerUnpacker:
    SUPPORTED_TYPES = tuple([dict, pd.DataFrame])

    @staticmethod
    def validate_sizes(unpacked_data):
        for row in unpacked_data["data"]:
            if len(row) != len(unpacked_data["columns"]):
                raise ValueError(
                    "Sizes mismatch:\n"
                    f'{len(row)} != {len(unpacked_data["columns"])}\n'
                    f"{row}\n"
                    f'{unpacked_data["columns"]}'
                )

    @staticmethod
    def unpack_data(data, unpacker_cb):
        unpacked_data = unpacker_cb(data)
        PackerUnpacker.validate_sizes(unpacked_data)
        return unpacked_data

    @staticmethod
    def pack_data(data, packer_cb):
        packed_data = packer_cb(data)
        return packed_data

    @staticmethod
    def dict_unpacker(data: dict):
        unpacked_data = copy.deepcopy(data)
        if "summaryRow" not in unpacked_data.keys():
            unpacked_data["summaryRow"] = None

        return unpacked_data

    @staticmethod
    def pandas_unpacker(data: pd.DataFrame):
        data = data.replace({np.nan: None})
        # data = data.astype(object).replace(np.nan, "-") # TODO: replace None later

        unpacked_data = {
            "columns": data.columns.to_list(),
            "data": data.values.tolist(),
            "summaryRow": None,
        }
        return unpacked_data

    @staticmethod
    def dict_packer(data):
        packed_data = {"columns": data["columns"], "data": data["data"]}
        if "summaryRow" in data.keys() and data["summaryRow"] is not None:
            packed_data["summaryRow"] = data["summaryRow"]
        return packed_data

    @staticmethod
    def pandas_packer(data):
        packed_data = pd.DataFrame(data=data["data"], columns=data["columns"])
        return packed_data


DATATYPE_TO_PACKER = {
    pd.DataFrame: PackerUnpacker.pandas_packer,
    dict: PackerUnpacker.dict_packer,
}

DATATYPE_TO_UNPACKER = {
    pd.DataFrame: PackerUnpacker.pandas_unpacker,
    dict: PackerUnpacker.dict_unpacker,
}

"""
iris = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)
iris.insert(loc=0, column="index", value=np.arange(len(iris)))
table = sly.app.widgets.Table(data=iris, fixed_cols=1)

@table.click
def show_image(datapoint: sly.app.widgets.Table.ClickedDataPoint):
    print("Column name = ", datapoint.column_name)
    print("Cell value = ", datapoint.cell_value)
    print("Row = ", datapoint.row)

"""


class Table(Widget):
    class Routes:
        CELL_CLICKED = "cell_clicked_cb"
        DOWNLOAD_AS_CSV = "download_as_csv"
        UPDATE_SORT = "update_sort_cb"

    class ClickedDataPoint:
        def __init__(self, column_index: int, column_name: str, cell_value: Any, row: dict, row_index: int = None):
            self.column_index = column_index
            self.column_name = column_name
            self.cell_value = cell_value
            self.row = row
            self.row_index = row_index
            self.button_name = None
            search = re.search(r"<button>(.*?)</button>", str(self.cell_value))
            if search is not None:
                self.button_name = search.group(1)

    def __init__(
        self,
        data=None,
        columns: Optional[list] = None,
        fixed_cols: Optional[int] = None,
        per_page: Optional[int] = 10,
        page_sizes: Optional[List[int]] = [10, 15, 30, 50, 100],
        width: Optional[str] = "auto",  # "200px", or "100%"
        sort_column_id: int = 0,
        sort_direction: Optional[Literal["asc", "desc"]] = "asc",
        widget_id: Optional[str] = None,
    ):
        """
        :param data: Data of table in different formats:
        1. Pandas Dataframe or pd.DataFrame(data=data, columns=columns)
        2. Python dict with structure {
                                        'columns': ['col_name_1', 'col_name_2', ...],
                                        'data': [
                                                    ['row_1_column_1', 'row_1_column_2', ...],
                                                    ['row_2_column_1', 'row_2_column_2', ...],
                                                    ...
                                                ]
                                      }
        """

        self._supported_types = PackerUnpacker.SUPPORTED_TYPES

        self._parsed_data = None
        self._data_type = None
        self._click_handled = False

        self._update_table_data(input_data=pd.DataFrame(data=data, columns=columns))

        self._per_page = per_page
        self._page_sizes = page_sizes
        self._fix_columns = fixed_cols
        self._width = width
        self._loading = False

        self._sort_column_id = (
            sort_column_id if self._validate_sort(column_id=sort_column_id) else 0
        )
        self._sort_direction = (
            sort_direction if self._validate_sort(direction=sort_direction) else "asc"
        )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "table_data": self._parsed_data,
            "table_options": {
                "perPage": self._per_page,
                "pageSizes": self._page_sizes,
                "fixColumns": self._fix_columns,
                "customDownloadUrl": None,
            },
            "loading": self._loading,
        }

    def get_json_state(self):
        return {
            "selected_row": {},
            "sort": {
                "columnIdx": self._sort_column_id,
                "direction": self._sort_direction,
            },
        }

    def _update_table_data(self, input_data):
        if input_data is not None:
            self._parsed_data = copy.deepcopy(self._get_unpacked_data(input_data=input_data))
        else:
            self._parsed_data = {"columns": [], "data": [], "summaryRow": None}
            self._data_type = dict

    def _get_packed_data(self, input_data, data_type):
        return PackerUnpacker.pack_data(data=input_data, packer_cb=DATATYPE_TO_PACKER[data_type])

    def _get_unpacked_data(self, input_data):
        input_data_type = type(input_data)

        if input_data_type not in self._supported_types:
            raise TypeError(
                f"Cannot parse input data, please use one of supported datatypes: {self._supported_types}\n"
                """
                            1. Pandas Dataframe \n
                            2. Python dict with structure {
                                        'columns': ['col_name_1', 'col_name_2', ...],
                                        'data': [
                                                            ['row_1_column_1', 'row_1_column_2', ...],
                                                            ['row_2_column_1', 'row_2_column_2', ...],
                                                            ...
                                                          ]
                                      }
                            """
            )

        return PackerUnpacker.unpack_data(
            data=input_data, unpacker_cb=DATATYPE_TO_UNPACKER[input_data_type]
        )

    @property
    def fixed_columns_num(self) -> int:
        return self._fix_columns

    @fixed_columns_num.setter
    def fixed_columns_num(self, value: int):
        self._fix_columns = value
        DataJson()[self.widget_id]["table_options"]["fixColumns"] = self._fix_columns

    @property
    def summary_row(self) -> List[Any]:
        if "summaryRow" not in self._parsed_data.keys():
            return None
        return self._parsed_data["summaryRow"]

    @summary_row.setter
    def summary_row(self, value: List[Any]):
        cols_num = len(self._parsed_data["columns"])
        if len(value) < cols_num:
            value.extend([""] * (cols_num - len(value)))
        elif len(value) > cols_num:
            value = value[:cols_num]
        DataJson()[self.widget_id]["table_data"]["summaryRow"] = value

    def to_json(self) -> Dict:
        return self._get_packed_data(self._parsed_data, dict)

    def to_pandas(self) -> pd.DataFrame:
        return self._get_packed_data(self._parsed_data, pd.DataFrame)

    def read_json(self, value: dict) -> None:
        self._update_table_data(input_data=value)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()
        self.clear_selection()

    def read_pandas(self, value: pd.DataFrame) -> None:
        self._update_table_data(input_data=value)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()
        self.clear_selection()

    def insert_row(self, data, index=-1):
        PackerUnpacker.validate_sizes({"columns": self._parsed_data["columns"], "data": [data]})

        table_data = self._parsed_data["data"]
        index = len(table_data) if index > len(table_data) or index < 0 else index

        self._parsed_data["data"].insert(index, data)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()

    def pop_row(self, index=-1):
        index = (
            len(self._parsed_data["data"]) - 1
            if index > len(self._parsed_data["data"]) or index < 0
            else index
        )

        if len(self._parsed_data["data"]) != 0:
            popped_row = self._parsed_data["data"].pop(index)
            DataJson()[self.widget_id]["table_data"] = self._parsed_data
            DataJson().send_changes()
            return popped_row

    def get_selected_cell(self, state):
        # logger.debug(
        #     "Selected row",
        #     extra={"selected_row": state[self.widget_id]["selected_row"]},
        # )
        row_index = state[self.widget_id]["selected_row"].get("selectedRow")
        column_name = state[self.widget_id]["selected_row"].get("selectedColumnName")
        column_index = state[self.widget_id]["selected_row"].get("selectedColumn")
        row = state[self.widget_id]["selected_row"].get("selectedRowData")

        if row_index is None or column_name is None or column_index is None or row is None:
            # click table header or clear selection
            return None

        return {
            "column_index": column_index,
            "column_name": column_name,
            "row": row,
            "row_index": row_index,
            "cell_value": row[column_name],
        }

    def download_as_csv(self, func):
        route_path = self.get_route_path(Table.Routes.DOWNLOAD_AS_CSV)
        server = self._sly_app.get_server()
        DataJson()[self.widget_id]["table_options"]["customDownloadUrl"] = route_path

        @server.get(route_path)
        def _click():
            try:
                df = func()
                df_name = df.index.name
                if df_name is None:
                    df_name = "table"
                stream = io.StringIO()
                df.to_csv(stream, index=False)
                response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
                response.headers["Content-Disposition"] = f"attachment; filename={df_name}.csv"
                return response

            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

        return _click

    def click(self, func):
        route_path = self.get_route_path(Table.Routes.CELL_CLICKED)
        update_sorting_column_route_path = self.get_route_path(Table.Routes.UPDATE_SORT)
        server = self._sly_app.get_server()

        @server.post(update_sorting_column_route_path)
        def _update_sorting_column():
            StateJson().send_changes()
            return

        self._click_handled = True

        @server.post(route_path)
        def _click():
            try:
                value_dict = self.get_selected_cell(StateJson())
                if value_dict is None:
                    return
                datapoint = Table.ClickedDataPoint(**value_dict)
                func(datapoint)
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

        return _click

    @staticmethod
    def get_html_text_as_button(title="preview"):
        return f"<button>{title}</button>"

    @staticmethod
    def create_button(title) -> str:
        return f"<button>{title}</button>"

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    def clear_selection(self):
        StateJson()[self.widget_id]["selected_row"] = {}
        StateJson().send_changes()

    def delete_row(self, key_column_name, key_cell_value):
        col_index = self._parsed_data["columns"].index(key_column_name)
        row_indices = []
        for idx, row in enumerate(self._parsed_data["data"]):
            if row[col_index] == key_cell_value:
                row_indices.append(idx)
        if len(row_indices) == 0:
            raise ValueError('Column "{key_column_name}" does not have value "{key_cell_value}"')
        if len(row_indices) > 1:
            raise ValueError(
                'Column "{key_column_name}" has multiple cells with the value "{key_cell_value}". Value has to be unique'
            )
        self.pop_row(row_indices[0])

    def update_cell_value(self, key_column_name, key_cell_value, column_name, new_value):
        key_col_index = self._parsed_data["columns"].index(key_column_name)
        row_indices = []
        for idx, row in enumerate(self._parsed_data["data"]):
            if row[key_col_index] == key_cell_value:
                row_indices.append(idx)
        if len(row_indices) == 0:
            raise ValueError('Column "{key_column_name}" does not have value "{key_cell_value}"')
        if len(row_indices) > 1:
            raise ValueError(
                'Column "{key_column_name}" has multiple cells with the value "{key_cell_value}". Value has to be unique'
            )

        col_index = self._parsed_data["columns"].index(column_name)
        self._parsed_data["data"][row_indices[0]][col_index] = new_value
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()

    def update_matching_cells(self, key_column_name, key_cell_value, column_name, new_value):
        key_col_index = self._parsed_data["columns"].index(key_column_name)
        row_indices = []
        for idx, row in enumerate(self._parsed_data["data"]):
            if row[key_col_index] == key_cell_value:
                row_indices.append(idx)

        col_index = self._parsed_data["columns"].index(column_name)
        for row_idx in row_indices:
            self._parsed_data["data"][row_idx][col_index] = new_value
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()

    def sort(self, column_id: int = None, direction: Optional[Literal["asc", "desc"]] = None):
        self._validate_sort(column_id, direction)
        if column_id is not None:
            self._sort_column_id = column_id
            StateJson()[self.widget_id]["sort"]["columnIdx"] = column_id
        if direction is not None:
            self._sort_direction = direction
            StateJson()[self.widget_id]["sort"]["direction"] = direction
        StateJson().send_changes()

    def _validate_sort(
        self, column_id: int = None, direction: Optional[Literal["asc", "desc"]] = None
    ):
        if column_id is not None and type(column_id) is not int:
            raise ValueError(f'Incorrect value of "column_id": {type(column_id)} is not "int".')
        if column_id is not None and column_id < 0:
            raise ValueError(f'Incorrect value of "column_id": {column_id} < 0')
        if direction is not None and direction not in ["asc", "desc"]:
            raise ValueError(
                f'Incorrect value of "direction": {direction}. Value can be one of "asc" or "desc".'
            )
        return True
