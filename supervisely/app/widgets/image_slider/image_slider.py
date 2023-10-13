from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from copy import deepcopy
from typing import List

from supervisely import logger


class ImageSlider(Widget):
    def __init__(
        self,
        previews: List[str] = None,
        examples: List[List[str]] = None,
        combined_data: List[dict] = None,
        height: int = 200,
        selectable: bool = True,
        preview_idx: int = None,
        preview_url: str = None,
        widget_id: str = None,
    ):
        """
        Input parameters format examples:

            previews = ["https://i.imgur.com/1Ys222.png", "https://i.imgur.com/2Yj2QjQ.png",]

            examples = [["https://i.imgur.com/1Ys222.png", "https://i.imgur.com/3Yd243.png"], ...]

            combined_data = [
                {
                    "moreExamples": [
                        "https://i.imgur.com/1Ys222.png",
                        "https://i.imgur.com/3Yd243.png",
                    ],
                    "preview": "https://i.imgur.com/2Yj2QjQ.png"
                },
                ...
            ]
        """

        self._data, self._image_url_to_idx = self._process_data(previews, examples, combined_data)
        self._height = height
        self._selectable = selectable
        self._selected_idx = None
        self._selected = None

        if preview_idx is not None and preview_url is not None:
            raise ValueError("You can't specify both 'preview_idx' and 'preview_url'.")
        if preview_idx is not None:
            self._update_selected_by_idx(preview_idx)
        elif preview_url is not None:
            self._update_selected_by_url(preview_url)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _process_data(
        self,
        previews: List[str] = None,
        examples: List[List[str]] = None,
        combined_data: List[dict] = None,
        start_idx: int = 0,
    ):
        data = []
        image_url_to_idx = {}
        if previews is not None and combined_data is not None:
            raise ValueError("You can't specify both 'previews' and 'combined_data'.")
        if previews is None and combined_data is None:
            logger.info("'previews' and 'combined_data' are not specified.")
            examples = []
            combined_data = []
        if previews is None and examples is not None:
            raise ValueError("You must specify 'previews' if you specify 'examples'.")
        if examples is not None and combined_data is not None:
            raise ValueError("You can't specify both 'examples' and 'combined_data'.")
        if examples is not None:
            if all([type(example) is not list for example in examples]):
                raise ValueError("Input examples must be a list of lists.")
            if len(examples) != len(previews):
                raise ValueError(
                    f"Length of 'examples' ({len(examples)}) must be equal to length of 'previews' ({len(previews)})."
                )
        if previews is not None:
            data = []
            if examples is None:
                examples = [[preview] for preview in previews]
            for idx, (preview, more_example) in enumerate(zip(previews, examples), start_idx):
                data.append(
                    {
                        "moreExamples": deepcopy(more_example),
                        "preview": preview,
                        "idx": idx,
                    }
                )
                image_url_to_idx[preview] = idx
        else:
            data = deepcopy(combined_data)
            for idx, item in enumerate(data, start_idx):
                image_url_to_idx[item["preview"]] = idx

        return data, image_url_to_idx

    def _update_selected_by_idx(self, idx: int):
        if type(idx) is not int:
            raise ValueError(f"Index must be an integer, not {type(idx)}")
        if 0 > idx >= len(self._data):
            raise ValueError(
                f'"Index {self._selected_idx} can`t be be greater than the length of input URLs list".'
            )
        self._selected_idx = idx
        self._selected = self._data[self._selected_idx]

        StateJson()[self.widget_id]["selected"] = self._selected
        StateJson().send_changes()

    def _update_selected_by_url(self, url: str):
        set_index = self._image_url_to_idx.get(url)
        if set_index is None:
            raise ValueError(f"There is no {url} url in input urls list")
        self._selected_idx = set_index
        self._selected = self._data[self._selected_idx]

        StateJson()[self.widget_id]["selected"] = self._selected
        StateJson().send_changes()

    def get_json_data(self):
        return {
            "data": self._data,
            "options": {
                "selectable": self._selectable,
                "height": f"{self._height}px",
            },
        }

    def get_json_state(self):
        return {"selected": self._selected}

    def get_selected_preview(self):
        selected = StateJson()[self.widget_id].get("selected")
        if selected is None:
            return None
        return selected["preview"]

    def set_selected_preview(self, value: str):
        self._update_selected_by_url(value)

    def get_selected_idx(self):
        selected = StateJson()[self.widget_id].get("selected")
        if selected is None:
            return None
        self._selected_idx = selected["idx"]
        return self._selected_idx

    def set_selected_idx(self, value: int):
        self._update_selected_by_idx(value)

    def get_selected_examples(self):
        selected = StateJson()[self.widget_id].get("selected")
        if selected is None:
            return None
        return selected["moreExamples"]

    @property
    def is_selectable(self):
        self._selectable = DataJson()[self.widget_id]["options"]["selectable"]
        return self._selectable

    def enable_selection(self):
        self._selectable = True
        DataJson()[self.widget_id]["options"]["selectable"] = self._selectable
        DataJson().send_changes()

    def disable_selection(self):
        self._selectable = False
        DataJson()[self.widget_id]["options"]["selectable"] = self._selectable
        DataJson().send_changes()

    def get_data_length(self):
        return len(self._data)

    def get_data(self):
        return self._data

    def set_data(
        self,
        previews: List[str] = None,
        examples: List[List[str]] = None,
        combined_data: List[dict] = None,
    ):
        self._data, self._image_url_to_idx = self._process_data(previews, examples, combined_data)
        DataJson()[self.widget_id]["data"] = self._data
        DataJson().send_changes()
        self._selected_idx = None
        self._selected = None
        StateJson()[self.widget_id]["selected"] = self._selected
        StateJson().send_changes()

    def append_data(
        self,
        previews: List[str] = None,
        examples: List[List[str]] = None,
        combined_data: List[dict] = None,
    ):
        data, url_to_idx = self._process_data(previews, examples, combined_data, len(self._data))
        self._data.extend(data)
        self._image_url_to_idx.update(url_to_idx)
        DataJson()[self.widget_id]["data"] = self._data
        DataJson().send_changes()
