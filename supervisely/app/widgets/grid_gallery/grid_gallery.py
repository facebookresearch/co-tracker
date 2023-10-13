import copy
import uuid
import time

import supervisely
from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from typing import List, Optional
from supervisely.app.content import StateJson


class GridGallery(Widget):
    class Routes:
        IMAGE_CLICKED = "image_clicked_cb"

    def __init__(
        self,
        columns_number: int,
        annotations_opacity: float = 0.5,
        show_opacity_slider: bool = True,
        enable_zoom: bool = False,
        resize_on_zoom: bool = False,
        sync_views: bool = False,
        fill_rectangle: bool = True,
        border_width: int = 3,
        show_preview: bool = False,
        view_height: Optional[int] = None,
        widget_id: str = None,
    ):
        self._data = []
        self._layout = []
        self._annotations = {}

        self.columns_number = columns_number

        self._last_used_column_index = 0
        self._project_meta: supervisely.ProjectMeta = None
        self._loading = False

        #############################
        # grid gallery settings
        self._show_preview: bool = True
        self._fill_rectangle: bool = fill_rectangle
        self._border_width: int = border_width

        self._show_opacity_header: bool = show_opacity_slider
        self._opacity: float = annotations_opacity
        self._enable_zoom: bool = enable_zoom
        self._sync_views: bool = sync_views
        self._resize_on_zoom: bool = resize_on_zoom
        self._show_preview: bool = show_preview
        self._views_bindings: list = []
        self._view_height: int = view_height
        #############################

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _generate_project_meta(self):
        objects_dict = dict()

        for cell_data in self._data:
            annotation: supervisely.Annotation = cell_data["annotation"]
            for label in annotation.labels:
                objects_dict[label.obj_class.name] = label.obj_class

        objects_list = list(objects_dict.values())
        objects_collection = (
            supervisely.ObjClassCollection(objects_list) if len(objects_list) > 0 else None
        )

        self._project_meta = supervisely.ProjectMeta(obj_classes=objects_collection)
        return self._project_meta.to_json()

    def get_json_data(self):
        return {
            "content": {
                "projectMeta": self._generate_project_meta(),
                "layout": self._layout,
                "annotations": self._annotations,
            },
            "loading": self._loading,
        }

    def get_json_state(self):
        return {
            "options": {
                "showOpacityInHeader": self._show_opacity_header,
                "opacity": self._opacity,
                "enableZoom": self._enable_zoom,
                "syncViews": self._sync_views,
                "syncViewsBindings": self._views_bindings,
                "resizeOnZoom": self._resize_on_zoom,
                "fillRectangle": self._fill_rectangle,
                "borderWidth": self._border_width,
                "selectable": False,
                "viewHeight": self._view_height,
                "showPreview": self._show_preview,
            },
            "selectedImage": None,
            "activeFigure": None,
        }

    def get_column_index(self, incoming_value):
        if incoming_value is not None and 0 > incoming_value > self.columns_number:
            raise ValueError(f"column index == {incoming_value} is out of bounds")

        if incoming_value is None:
            incoming_value = self._last_used_column_index
            self._last_used_column_index = (self._last_used_column_index + 1) % self.columns_number
        else:
            self._last_used_column_index = incoming_value

        return incoming_value

    def append(
        self,
        image_url: str,
        annotation: supervisely.Annotation = None,
        title: str = "",
        column_index: int = None,
        zoom_to: int = None,
        zoom_factor: float = 1.2,
        title_url=None,
    ):
        column_index = self.get_column_index(incoming_value=column_index)
        cell_uuid = str(
            uuid.uuid5(
                namespace=uuid.NAMESPACE_URL,
                name=f"{image_url}_{title}_{column_index}_{time.time()}",
            ).hex
        )

        self._data.append(
            {
                "image_url": image_url,
                "annotation": supervisely.Annotation((1, 1))
                if annotation is None
                else annotation.clone(),
                "column_index": column_index,
                "title": title
                if title_url is None
                else title + ' <i class="zmdi zmdi-open-in-new"></i>',
                "cell_uuid": cell_uuid,
                "zoom_to": zoom_to,
                "zoom_factor": zoom_factor,
                "title_url": title_url,
            }
        )

        self._update()
        return cell_uuid

    def clean_up(self):
        self._data = []
        self._layout = []
        self._annotations = {}
        self._update()
        self.update_data()

    def _update_layout(self):
        layout = [[] for _ in range(self.columns_number)]

        for cell_data in self._data:
            layout[cell_data["column_index"]].append(cell_data["cell_uuid"])

        self._layout = copy.deepcopy(layout)
        DataJson()[self.widget_id]["content"]["layout"] = self._layout

    def _update_annotations(self):
        annotations = {}

        for cell_data in self._data:
            annotations[cell_data["cell_uuid"]] = {
                "url": cell_data["image_url"],
                "figures": [label.to_json() for label in cell_data["annotation"].labels],
                "title": cell_data["title"],
                "title_url": cell_data["title_url"],
            }
            if not cell_data["zoom_to"] is None:
                zoom_params = {
                    "figureId": cell_data["zoom_to"],
                    "factor": cell_data["zoom_factor"],
                }
                annotations[cell_data["cell_uuid"]]["zoomToFigure"] = zoom_params

        self._annotations = copy.deepcopy(annotations)
        DataJson()[self.widget_id]["content"]["annotations"] = self._annotations

    def _update_project_meta(self):
        DataJson()[self.widget_id]["content"]["projectMeta"] = self._generate_project_meta()

    def _update(self):
        self._update_layout()
        self._update_annotations()
        self._update_project_meta()

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    def sync_images(self, image_ids: List[str]):
        self._views_bindings.append(image_ids)
        StateJson()[self.widget_id]["options"]["syncViewsBindings"] = self._views_bindings
        StateJson().send_changes()
