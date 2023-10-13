import os
import uuid
from pathlib import Path
from urllib3.util import parse_url
from typing import List, Optional, Tuple

import supervisely as sly
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Button, FolderThumbnail, GridGallery, Slider, Widget
from supervisely.app.fastapi.offline import get_offline_session_files_path


class ImagePairSequence(Widget):
    def __init__(
        self,
        opacity: Optional[float] = 0.4,
        enable_zoom: Optional[bool] = False,
        sync_views: Optional[bool] = True,
        slider_title: Optional[str] = "pairs",
        widget_id=None,
    ):
        """NOTE: The `path` argument can be either the path from the local `static_dir` or the URL to the image."""

        self._api = sly.Api.from_env()
        self._team_id = sly.env.team_id()

        # init data for gallery
        self._left_data = []
        self._right_data = []

        # init gallery options
        self._columns = 2
        self._current_grid = 0
        self._total_grids = 0
        self._slider_title = slider_title
        self._need_update = False

        self._first_button = Button(
            "", icon="zmdi zmdi-skip-previous", plain=True, button_size="mini"
        )
        self._prev_button = Button(
            "", icon="zmdi zmdi-chevron-left", plain=True, button_size="mini"
        )
        self._next_button = Button(
            "", icon="zmdi zmdi-chevron-right", plain=True, button_size="mini"
        )
        self._last_button = Button("", icon="zmdi zmdi-skip-next", plain=True, button_size="mini")

        self._slider = Slider(show_stops=True, min=1, max=1, step=1, value=1)

        self._folder_thumbnail = FolderThumbnail()

        self._grid_gallery = GridGallery(
            columns_number=self._columns,
            annotations_opacity=opacity,
            show_opacity_slider=True,
            enable_zoom=enable_zoom,
            resize_on_zoom=False,
            sync_views=sync_views,
            fill_rectangle=False,
        )
        self._grid_gallery.hide()

        @self._slider.value_changed
        def on_slider_change(value):
            self._update_gallery(int(value))

        @self._first_button.click
        def on_first_button_click():
            if self._current_grid > 1:
                self._update_gallery(1)

        @self._prev_button.click
        def on_prev_button_click():
            if self._current_grid > 1:
                self._update_gallery(self._current_grid - 1)

        @self._next_button.click
        def on_next_button_click():
            if self._current_grid < self._total_grids:
                self._update_gallery(self._current_grid + 1)

        @self._last_button.click
        def on_last_button_click():
            if self._current_grid < self._total_grids:
                self._update_gallery(self._total_grids)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> dict:
        return {
            "leftAnnotations": [],
            "rightAnnotations": [],
        }

    def get_json_state(self) -> dict:
        return {
            "currentGrid": self._current_grid,
            "totalGrids": self._total_grids,
            "sliderTitle": f" {self._slider_title}",
        }

    def append_left(self, path: str, ann: sly.Annotation = None, title: str = None):
        self._add_with_check("left", [[path, ann, title]])
        self._update_data()

    def append_right(self, path: str, ann: sly.Annotation = None, title: str = None):
        self._add_with_check("right", [[path, ann, title]])
        self._update_data()

    def extend_left(
        self, paths: List[str], anns: List[sly.Annotation] = None, titles: List[str] = None
    ):
        anns = [None] * len(paths) if anns is None else anns
        titles = [None] * len(paths) if titles is None else titles
        data = list(zip(paths, anns, titles))
        self._add_with_check("left", data)
        self._update_data()

    def extend_right(
        self, paths: List[str], anns: List[sly.Annotation] = None, titles: List[str] = None
    ):
        anns = [None] * len(paths) if anns is None else anns
        titles = [None] * len(paths) if titles is None else titles
        data = list(zip(paths, anns, titles))
        self._add_with_check("right", data)
        self._update_data()

    def append_pair(
        self,
        left: Tuple[str, Optional[sly.Annotation], Optional[str]],
        right: Tuple[str, Optional[sly.Annotation], Optional[str]],
    ):
        self._add_with_check("left", [left])
        self._add_with_check("right", [right])
        self._update_data()

    def extend_pairs(self, left: List[Tuple], right: List[Tuple]):
        self._add_with_check("left", left)
        self._add_with_check("right", right)
        self._update_data()

    def clean_up(self):
        self._left_data = []
        self._right_data = []
        self._grid_gallery.hide()
        self._slider.set_value(1)
        self._slider.set_max(1)
        self._need_update = False
        self._current_grid = 0
        self._total_grids = 0
        StateJson()[self.widget_id]["totalGrids"] = 0
        StateJson()[self.widget_id]["currentGrid"] = 0
        StateJson().send_changes()

    def _check_paths(self, paths):
        for path in paths:
            parsed_path = parse_url(path)
            if parsed_path.scheme not in (None, "http", "https"):
                raise ValueError(f"Invalid path or url to image: {path}")
            
    def _prepare_annotations(self, anns):
        new_anns = []
        for ann in anns:
            if ann is not None:
                new_labels = []
                for label in ann.labels:
                    if label.geometry.sly_id is None:
                        label.geometry.sly_id = str(uuid.uuid4().int)
                    new_labels.append(label)
                ann = ann.clone(labels=new_labels)
            new_anns.append(ann)
        return new_anns

    def _add_with_check(self, side, data):
        paths, anns, titles = [], [], []
        for item in data:
            if len(item) < 3:
                item = item + (None,) * (3 - len(item))
            path, ann, title = item
            paths.append(path)
            anns.append(ann)
            titles.append(title)

        self._check_paths(paths)
        new_anns = self._prepare_annotations(anns)

        total_grids = max(len(self._left_data), len(self._right_data), 1)
        if self._total_grids != total_grids:
            self._need_update = True

        has_empty_before = any([len(self._left_data) == 0, len(self._right_data) == 0])

        ann_jsons = [a.to_json() if a is not None else None for a in new_anns]
        data = list(zip(paths, new_anns, titles))
        if side == "left":
            self._left_data.extend(data)
            DataJson()[self.widget_id]["leftAnnotations"].extend(ann_jsons)
        elif side == "right":
            self._right_data.extend(data)
            DataJson()[self.widget_id]["rightAnnotations"].extend(ann_jsons)
        DataJson().send_changes()

        has_empty_after = any([len(self._left_data) == 0, len(self._right_data) == 0])
        if has_empty_before and not has_empty_after:
            self._need_update = True

        self._dump_image_to_offline_sessions_file(paths)

    def _dump_image_to_offline_sessions_file(self, paths: List[str]):
        if sly.is_production():
            remote_dir = get_offline_session_files_path(self._api.task_id)
            remote_dir = remote_dir.joinpath("sly", "css", "app", "widgets", "image_pair_sequence")
            dst_paths = [remote_dir.joinpath(Path(path).name).as_posix() for path in paths]
            local_paths = [self._download_image(path) for path in paths]

            self._api.file.upload_bulk(
                team_id=self._team_id,
                src_paths=local_paths,
                dst_paths=dst_paths,
            )

    def _download_image(self, path: str):
        if path.lstrip("/").startswith("static/"):
            path = path.lstrip("/")[len("static/") :]
            app = sly.Application()
            save_path = os.path.join(app.get_static_dir(), path)
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"File {save_path} not found")
        else:
            save_path = os.path.join(sly.app.get_data_dir(), sly.fs.get_file_name_with_ext(path))
            sly.fs.download(path, save_path)
        return save_path

    def _update_data(self):
        self._total_grids = max(len(self._left_data), len(self._right_data), 1)
        min_len = min(len(self._left_data), len(self._right_data))
        if self._slider.get_value() == self._slider.get_max():
            self._need_update = True

        self._slider.set_max(self._total_grids)

        if self._need_update and min_len > 0:
            self._update_gallery(self._total_grids)

        StateJson()[self.widget_id]["totalGrids"] = self._total_grids
        StateJson().send_changes()

    def _update_gallery(self, page: int):
        if self._grid_gallery.is_hidden():
            self._grid_gallery.show()
        self._slider.set_value(page)
        self._current_grid = page
        StateJson()[self.widget_id]["currentGrid"] = page
        StateJson().send_changes()

        self._grid_gallery.clean_up()

        len_left = len(self._left_data)
        len_right = len(self._right_data)

        if len_left > 0 and len_right > 0:
            left = self._left_data[page - 1] if page <= len_left else self._left_data[-1]
            self._grid_gallery.append(*left)  # set left

            right = self._right_data[page - 1] if page <= len_right else self._right_data[-1]
            self._grid_gallery.append(*right)  # set right

        DataJson().send_changes()
        self._need_update = False

    def disable(self):
        self._disabled = True
        self._grid_gallery.disable()
        self._first_button.disable()
        self._prev_button.disable()
        self._next_button.disable()
        self._last_button.disable()
        self._slider.disable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._disabled = False
        self._grid_gallery.enable()
        self._first_button.enable()
        self._prev_button.enable()
        self._next_button.enable()
        self._last_button.enable()
        self._slider.enable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()
