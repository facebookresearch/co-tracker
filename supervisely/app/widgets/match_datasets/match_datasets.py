from typing import List, Union
from supervisely import Api, DatasetInfo
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget


class MatchDatasets(Widget):
    def __init__(
        self,
        left_datasets: List[DatasetInfo] = None,
        right_datasets: List[DatasetInfo] = None,
        left_name=None,
        right_name=None,
        widget_id=None,
    ):
        self._left_ds = left_datasets
        self._right_ds = right_datasets
        self._left_name = "Left Datasets" if left_name is None else left_name
        self._right_name = "Right Datasets" if right_name is None else right_name
        self._api = Api()

        self._done = False

        self._results = None
        self._stat = {}

        super().__init__(widget_id=widget_id, file_path=__file__)

        if self._left_ds is not None and self._right_ds is not None:
            self._load_datasets_statistic()

    def get_json_data(self):
        return {"left_name": self._left_name, "right_name": self._right_name}

    def get_json_state(self):
        return {"done": self._done}

    def set(
        self,
        left_datasets: List[DatasetInfo] = None,
        right_datasets: List[DatasetInfo] = None,
        left_name=None,
        right_name=None,
    ):
        self._left_ds = left_datasets
        self._right_ds = right_datasets
        self._left_name = self._left_name if left_name is None else left_name
        self._right_name = self._right_name if right_name is None else right_name

        self._done = False
        self._loading = False

        self._results = None
        self._stat = {}

        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()
        StateJson()[self.widget_id] = self.get_json_state()
        StateJson().send_changes()

        if self._left_ds is not None and self._right_ds is not None:
            self._load_datasets_statistic()

    def get_stat(self):
        return self._stat

    def _load_datasets_statistic(self):
        self._stat = {}
        ds_info1, ds_images1 = self._get_all_images(self._left_ds)
        ds_info2, ds_images2 = self._get_all_images(self._right_ds)
        result = self._process_items(ds_info1, ds_images1, ds_info2, ds_images2)

        DataJson()[self.widget_id]["table"] = result
        DataJson().send_changes()
        StateJson()[self.widget_id]["done"] = True
        StateJson().send_changes()

    def _process_items(self, ds_info1, collection1, ds_info2, collection2):
        ds_names = ds_info1.keys() | ds_info2.keys()

        results = []
        for idx, name in enumerate(ds_names):
            compare = {"dsIndex": idx}
            images1 = collection1.get(name, [])
            images2 = collection2.get(name, [])
            if len(images1) == 0:
                compare["message"] = ["unmatched (in GT project)"]
                compare["icon"] = [
                    ["zmdi zmdi-long-arrow-left", "zmdi zmdi-alert-circle-o"]
                ]

                compare["color"] = ["#F39C12"]
                compare["numbers"] = [-1]
                compare["left"] = {"name": ""}
                compare["right"] = {"name": name, "count": len(images2)}
                self._stat[name] = {"dataset_matched": "right"}
            elif len(images2) == 0:
                compare["message"] = ["unmatched (in PRED project)"]
                compare["icon"] = [
                    ["zmdi zmdi-alert-circle-o", "zmdi zmdi-long-arrow-right"]
                ]
                compare["color"] = ["#F39C12"]
                compare["numbers"] = [-1]
                compare["left"] = {"name": name, "count": len(images1)}
                compare["right"] = {"name": ""}
                self._stat[name] = {"dataset_matched": "left"}
            else:
                img_dict1 = {img_info.name: img_info for img_info in images1}
                img_dict2 = {img_info.name: img_info for img_info in images2}
                matched = []
                diff = []
                same_names = img_dict1.keys() & img_dict2.keys()
                for img_name in same_names:
                    dest = (
                        matched
                        if img_dict1[img_name].hash == img_dict2[img_name].hash
                        else diff
                    )
                    dest.append(
                        {"left": img_dict1[img_name], "right": img_dict2[img_name]}
                    )

                uniq1 = [img_dict1[name] for name in img_dict1.keys() - same_names]
                uniq2 = [img_dict2[name] for name in img_dict2.keys() - same_names]

                compare["message"] = [
                    "matched",
                    "conflicts",
                    "unique (left)",
                    "unique (right)",
                ]
                compare["icon"] = [
                    ["zmdi zmdi-check"],
                    ["zmdi zmdi-close"],
                    ["zmdi zmdi-plus-circle-o"],
                    ["zmdi zmdi-plus-circle-o"],
                ]
                compare["color"] = ["green", "red", "#20a0ff", "#20a0ff"]
                compare["numbers"] = [len(matched), len(diff), len(uniq1), len(uniq2)]
                compare["left"] = {"name": name, "count": len(images1)}
                compare["right"] = {"name": name, "count": len(images2)}

                self._stat[name] = {
                    "dataset_matched": "both",
                    "matched": matched,
                    "conflicts": diff,
                    "unique_left": uniq1,
                    "unique_right": uniq2,
                }
            results.append(compare)

        self._results = results
        return results

    def _get_all_images(self, datasets):
        ds_info = {}
        ds_images = {}
        for dataset in datasets:
            ds_info[dataset.name] = dataset
            images = self._api.image.get_list(dataset.id)
            ds_images[dataset.name] = images
        return (
            ds_info,
            ds_images,
        )
