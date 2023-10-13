from typing import Union
import supervisely as sly
from supervisely.project.project_meta import ProjectMeta
from supervisely.api.api import Api
from supervisely.annotation.annotation import Annotation


class Gallery:
    def __init__(self, task_id, api: Api, v_model, project_meta: ProjectMeta, col_number: int, preview_info=False,
                 enable_zoom=False, resize_on_zoom=False, sync_views=False, show_preview=True, selectable=False,
                 opacity=0.5, show_opacity_header=True, fill_rectangle=False, border_width=3):
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        self._project_meta = project_meta.clone()
        self._data = {}
        self.col_number = col_number
        self.preview_info = preview_info
        self._need_zoom = False
        self._with_title_url = False
        if not isinstance(self.col_number, int):
            raise ValueError("Columns number must be integer, not {}".format(type(self.col_number).__name__))

        self._options = {
            "enableZoom": enable_zoom,
            "syncViews": sync_views,
            "resizeOnZoom": resize_on_zoom,
            "showPreview": show_preview,
            "selectable": selectable,
            "opacity": opacity,
            "showOpacityInHeader": show_opacity_header,
            "fillRectangle": fill_rectangle,
            "borderWidth": border_width
        }
        self._options_initialized = False

    def add_item(self, title, image_url, ann: Union[Annotation, dict] = None, col_index=None, custom_info: dict = None,
                 zoom_to_figure=None, title_url=None):

        if col_index is not None:
            if col_index <= 0 or col_index > self.col_number:
                raise ValueError("Column number is not correct, check your input data")

        res_ann = Annotation((1, 1))
        if ann is not None:
            if type(ann) is dict:
                res_ann = Annotation.from_json(ann, self._project_meta)
            else:
                res_ann = ann.clone()

        self._data[title] = {"image_url": image_url, "ann": res_ann, "col_index": col_index}

        if zoom_to_figure is not None:
            self._data[title]["zoom_to_figure"] = zoom_to_figure
            self._need_zoom = True

        if title_url is not None:
            self.preview_info = True
            self._with_title_url = True
            self._data[title]["labelingUrl"] = title_url

        if self.preview_info:
            if custom_info is not None:
                self._data[title]["info"] = custom_info
            else:
                self._data[title]["info"] = None

    def add_item_by_id(self, image_id, with_ann=True, col_index=None, info_dict=None,
                 zoom_to_figure=None, title_url=None):
        image_info = self._api.image.get_info_by_id(image_id)
        if with_ann:
            ann_info = self._api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, self._project_meta)
        else:
            ann = None

        self.add_item(image_info.name, image_info.full_storage_url, ann, col_index, info_dict, zoom_to_figure, title_url)

    def _get_item_annotation(self, name):
        if self.preview_info:
            if self._with_title_url:
                return {
                    "url": self._data[name]["image_url"],
                    "figures": [label.to_json() for label in self._data[name]["ann"].labels],
                    "title": name,
                    "info": self._data[name]["info"],
                    "labelingUrl": self._data[name]["labelingUrl"]
                }
            else:
                return {
                    "url": self._data[name]["image_url"],
                    "figures": [label.to_json() for label in self._data[name]["ann"].labels],
                    "title": name,
                    "info": self._data[name]["info"]
                }
        else:
            return {
                "url": self._data[name]["image_url"],
                "figures": [label.to_json() for label in self._data[name]["ann"].labels],
                "title": name,
            }

    def update(self, options=True):
        if len(self._data) == 0:
            raise ValueError("Items list is empty")

        gallery_json = self.to_json()

        if options is True or self._options_initialized is False:
            if self._need_zoom:
                self._options["resizeOnZoom"] = True
            self._api.task.set_field(self._task_id, self._v_model, gallery_json)
            self._options_initialized = True
        else:
            self._api.task.set_field(self._task_id, f"{self._v_model}.content", gallery_json["content"])

    def _zoom_to_figure(self, annotations):
        items = self._data.items()
        zoom_to_figure_name = "zoomToFigure"
        for item in items:
            curr_image_name = item[0]
            curr_image_data = item[1]

            if type(curr_image_data["zoom_to_figure"]) is not tuple:
                raise ValueError("Option zoom_to_figure not set for {} image".format(curr_image_name))
            elif type(curr_image_data["zoom_to_figure"]) is None:
                raise ValueError("Option zoom_to_figure not set for {} image".format(curr_image_name))

            zoom_params = {
                "figureId": curr_image_data["zoom_to_figure"][0],
                "factor": curr_image_data["zoom_to_figure"][1]
            }
            annotations[curr_image_name][zoom_to_figure_name] = zoom_params

    def _add_info(self, annotations):
        items = self._data.items()
        for item in items:
            curr_image_name = item[0]
            curr_image_data = item[1]

            annotations[curr_image_name]["info"] = curr_image_data["info"]

    def to_json(self):
        annotations = {}
        layout = [[] for _ in range(self.col_number)]
        index_in_layout = 0

        for curr_image_name, curr_image_data in self._data.items():
            annotations[curr_image_name] = self._get_item_annotation(curr_image_name)

            curr_col_index = curr_image_data["col_index"]
            if curr_col_index is not None:
                layout[curr_col_index - 1].append(curr_image_name)
            else:
                if index_in_layout == self.col_number:
                    index_in_layout = 0
                layout[index_in_layout].append(curr_image_name)
                index_in_layout += 1

        if self._need_zoom:
            self._zoom_to_figure(annotations)

        if self.preview_info:
            self._add_info(annotations)

        return {
            "content": {
                "projectMeta": self._project_meta.to_json(),
                "layout": layout,
                "annotations": annotations
            },
            "options": {
                **self._options
            }
        }
