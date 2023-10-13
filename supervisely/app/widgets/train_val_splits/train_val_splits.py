import os
from typing import List, Optional, Dict, Union, Tuple
from supervisely import Project, Api
from supervisely.project.project import ItemInfo
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import (
    Widget,
    RadioTabs,
    Container,
    NotificationBox,
    SelectDataset,
    SelectString,
    Field,
    SelectTagMeta,
)
from supervisely.app.widgets.random_splits_table.random_splits_table import RandomSplitsTable
from supervisely.app import get_data_dir
from supervisely._utils import rand_str
from supervisely.io.fs import remove_dir
import supervisely as sly


class TrainValSplits(Widget):
    def __init__(
        self,
        project_id: Optional[int] = None,
        project_fs: Optional[Project] = None,
        random_splits: Optional[bool] = True,
        tags_splits: Optional[bool] = True,
        datasets_splits: Optional[bool] = True,
        widget_id: Optional[int] = None,
    ):
        self._project_id = project_id
        self._project_fs: Project = project_fs
        if project_fs is not None and project_id is not None:
            raise ValueError(
                "You can not provide both project_id and project_fs parameters to TrainValSplits widget."
            )
        if project_fs is None and project_id is None:
            raise ValueError(
                "You should provide at least one of: project_id or project_fs parameters to TrainValSplits widget."
            )

        self._project_info = None
        if project_id is not None:
            self._api = Api()
            self._project_info = self._api.project.get_info_by_id(self._project_id)
        self._random_splits_table: RandomSplitsTable = None
        self._train_tag_select: SelectTagMeta = None
        self._val_tag_select: SelectTagMeta = None
        self._untagged_select: SelectString = None
        self._train_ds_select: Union[SelectDataset, SelectString] = None
        self._val_ds_select: Union[SelectDataset, SelectString] = None
        self._split_methods = []
        contents = []
        tabs_descriptions = []
        if random_splits:
            self._split_methods.append("Random")
            tabs_descriptions.append("Shuffle data and split with defined probability")
            contents.append(self._get_random_content())
        if tags_splits:
            self._split_methods.append("Based on item tags")
            tabs_descriptions.append("Images should have assigned train or val tag")
            contents.append(self._get_tags_content())
        if datasets_splits:
            self._split_methods.append("Based on datasets")
            tabs_descriptions.append("Select one or several datasets for every split")
            contents.append(self._get_datasets_content())
        if not self._split_methods:
            raise ValueError(
                "Any of split methods [random_splits, tags_splits, datasets_splits] must be specified in TrainValSplits."
            )

        self._content = RadioTabs(
            titles=self._split_methods,
            descriptions=tabs_descriptions,
            contents=contents,
        )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_random_content(self):
        if self._project_id is not None:
            items_count = self._project_info.items_count
        elif self._project_fs is not None:
            items_count = self._project_fs.total_items
        self._random_splits_table = RandomSplitsTable(items_count)

        return Container(widgets=[self._random_splits_table], direction="vertical", gap=5)

    def _get_tags_content(self):
        notification_box = NotificationBox(
            title="Notice: How to make equal splits",
            description="Choose the same tag for train/validation to make splits equal. Can be used for debug and for tiny projects",
            box_type="info",
        )
        if self._project_id is not None:
            self._train_tag_select = SelectTagMeta(project_id=self._project_id, show_label=False)
            self._val_tag_select = SelectTagMeta(project_id=self._project_id, show_label=False)
        elif self._project_fs is not None:
            self._train_tag_select = SelectTagMeta(
                project_meta=self._project_fs.meta, show_label=False
            )
            self._val_tag_select = SelectTagMeta(
                project_meta=self._project_fs.meta, show_label=False
            )
        self._untagged_select = SelectString(
            values=["train", "val", "ignore"],
            labels=[
                "add untagged images to train set",
                "add untagged images to val set",
                "ignore untagged images",
            ],
            placeholder="Select action",
        )
        train_field = Field(
            self._train_tag_select,
            title="Train tag",
            description="all images with this tag are considered as training set",
        )
        val_field = Field(
            self._val_tag_select,
            title="Validation tag",
            description="all images with this tag are considered as validation set",
        )
        without_tags_field = Field(
            self._untagged_select,
            title="Images without selected tags",
            description="Choose what to do with untagged images",
        )
        return Container(
            widgets=[
                notification_box,
                train_field,
                val_field,
                without_tags_field,
            ],
            direction="vertical",
            gap=5,
        )

    def _get_datasets_content(self):
        notification_box = NotificationBox(
            title="Notice: How to make equal splits",
            description="Choose the same dataset(s) for train/validation to make splits equal. Can be used for debug and for tiny projects",
            box_type="info",
        )
        if self._project_id is not None:
            self._train_ds_select = SelectDataset(
                project_id=self._project_id, multiselect=True, compact=True, show_label=False
            )
            self._val_ds_select = SelectDataset(
                project_id=self._project_id, multiselect=True, compact=True, show_label=False
            )
        elif self._project_fs is not None:
            ds_names = [ds.name for ds in self._project_fs.datasets]
            self._train_ds_select = SelectString(ds_names, multiple=True)
            self._val_ds_select = SelectString(ds_names, multiple=True)
        train_field = Field(
            self._train_ds_select,
            title="Train dataset(s)",
            description="all images in selected dataset(s) are considered as training set",
        )
        val_field = Field(
            self._val_ds_select,
            title="Validation dataset(s)",
            description="all images in selected dataset(s) are considered as validation set",
        )
        return Container(
            widgets=[notification_box, train_field, val_field], direction="vertical", gap=5
        )

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def get_splits(self) -> Tuple[List[ItemInfo], List[ItemInfo]]:
        split_method = self._content.get_active_tab()
        tmp_project_dir = None
        if self._project_fs is None:
            tmp_project_dir = os.path.join(get_data_dir(), rand_str(15))
            Project.download(self._api, self._project_id, tmp_project_dir)
        project_dir = tmp_project_dir if tmp_project_dir is not None else self._project_fs.directory
        if split_method == "Random":
            splits_counts = self._random_splits_table.get_splits_counts()
            train_count = splits_counts["train"]
            val_count = splits_counts["val"]
            val_part = val_count / (val_count + train_count)
            project = Project(project_dir, sly.OpenMode.READ)
            n_images = project.total_items
            new_val_count = round(val_part * n_images)
            new_train_count = n_images - new_val_count
            train_set, val_set = Project.get_train_val_splits_by_count(
                project_dir, new_train_count, new_val_count
            )

        elif split_method == "Based on item tags":
            train_tag_name = self._train_tag_select.get_selected_name()
            val_tag_name = self._val_tag_select.get_selected_name()
            add_untagged_to = self._untagged_select.get_value()
            train_set, val_set = Project.get_train_val_splits_by_tag(
                project_dir, train_tag_name, val_tag_name, add_untagged_to
            )

        elif split_method == "Based on datasets":
            if self._project_id is not None:
                self._train_ds_select: SelectDataset
                self._val_ds_select: SelectDataset
                train_ds_ids = self._train_ds_select.get_selected_ids()
                val_ds_ids = self._val_ds_select.get_selected_ids()
                ds_infos = self._api.dataset.get_list(self._project_id)
                train_ds_names, val_ds_names = [], []
                for ds_info in ds_infos:
                    if ds_info.id in train_ds_ids:
                        train_ds_names.append(ds_info.name)
                    if ds_info.id in val_ds_ids:
                        val_ds_names.append(ds_info.name)
            elif self._project_fs is not None:
                self._train_ds_select: SelectString
                self._val_ds_select: SelectString
                train_ds_names = self._train_ds_select.get_value()
                val_ds_names = self._val_ds_select.get_value()
            train_set, val_set = Project.get_train_val_splits_by_dataset(
                project_dir, train_ds_names, val_ds_names
            )

        if tmp_project_dir is not None:
            remove_dir(tmp_project_dir)
        return train_set, val_set

    def disable(self):
        self._content.disable()
        self._random_splits_table.disable()
        self._train_tag_select.disable()
        self._val_tag_select.disable()
        self._untagged_select.disable()
        self._train_ds_select.disable()
        self._val_ds_select.disable()
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._content.enable()
        self._random_splits_table.enable()
        self._train_tag_select.enable()
        self._val_tag_select.enable()
        self._untagged_select.enable()
        self._train_ds_select.enable()
        self._val_ds_select.enable()
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()
