from typing import Dict, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import (
    Widget,
    SelectDataset,
    Checkbox,
    Select,
    Text,
    generate_id,
)
from supervisely.api.api import Api
from supervisely.app.widgets.select_sly_utils import _get_int_or_env


class SelectItem(Widget):
    def __init__(
        self,
        dataset_id: int = None,
        compact: bool = True,
        show_label: bool = True,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
    ):
        self._api = Api()
        self._dataset_id = dataset_id
        self._compact = compact
        self._show_label = show_label
        self._size = size

        self._ds_selector = None
        self._limit_message = None

        super().__init__(widget_id=widget_id, file_path=__file__)

        items = []
        if self._dataset_id is not None:
            items = self._get_items_from_server(dataset_id)
            need_limit, ds_items_count = self._need_set_limit(dataset_id)
            if need_limit is True:
                self._set_limit(ds_items_count)

        self._item_selector = Select(
            items=items, filterable=True, size=size, widget_id=generate_id()
        )

        self._dataset_id = _get_int_or_env(self._dataset_id, "modal.state.slyDatasetId")
        if self._compact is True:
            if self._dataset_id is None:
                raise ValueError(
                    '"dataset_id" have to be passed as argument or "compact" has to be False'
                )
        else:
            self._show_label = True
            self._ds_selector = SelectDataset(
                default_id=self._dataset_id,
                multiselect=False,
                compact=self._compact,
                show_label=True,
                size=self._size,
                widget_id=generate_id(),
            )

            @self._ds_selector.value_changed
            def dataset_changed(dataset_id):
                # print("dataset_id ---> ", dataset_id)
                self.refresh_items(dataset_id)


    def get_json_data(self) -> Dict:
        res = {
            "limit_message": self._limit_message,
            "compact": self._compact,
            "show_label": self._show_label,
        }
        return res

    def get_json_state(self) -> Dict:
        res = {}
        return res

    def get_selected_id(self):
        return self._item_selector.get_value()

    def _need_set_limit(self, dataset_id: int = None, limit: int = 50):
        count = None
        if dataset_id is not None:
            dataset_info = self._api.dataset.get_info_by_id(dataset_id)
            count = dataset_info.items_count
            if count > limit:
                return True, count
        return False, count

    def _set_limit(self, dataset_items: int = None, limit: int = 50):
        self._limit_message = (
            f"Showed first {limit} of {dataset_items} items in dataset"
        )
        DataJson()[self.widget_id]["limit_message"] = self._limit_message
        DataJson().send_changes()

    def _get_items_from_server(self, dataset_id: int = None, limit: int = 50):
        items = []
        if dataset_id is not None:
            infos = self._api.image.get_list(dataset_id, limit=limit)
            items = [Select.Item(value=info.id, label=info.name) for info in infos]
        return items

    def refresh_items(self, dataset_id: int = None, limit: int = 50):
        items = self._get_items_from_server(dataset_id, limit)
        self._item_selector.set(items=items)
        need_limit, ds_items_count = self._need_set_limit(dataset_id)
        if need_limit is True:
            self._set_limit(ds_items_count)
