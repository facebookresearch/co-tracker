from typing import Union
from supervisely.project.project_meta import ProjectMeta
from supervisely.api.api import Api
from supervisely.annotation.annotation import Annotation


# html example
# <sly-grid-gallery
#         v-if="data.gallery"
#         :content="data.gallery.content"
#         :options="data.gallery.options">
# </sly-grid-gallery>


class SingleImageGallery:

    def __init__(self, task_id, api: Api, v_model, project_meta: ProjectMeta):
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        self._project_meta = project_meta.clone()
        self._image_url = None
        self._ann = None
        self._options_initialized = False
        self.set_options()

    def set_options(self, enable_zoom=True, show_preview=True, selectable=False, opacity=0.5,
                    show_opacity_in_header=True, fill_rectangle=False):
        self._options = {
            "enableZoom": enable_zoom,
            "showPreview": show_preview,
            "selectable": selectable,
            "opacity": opacity,
            "showOpacityInHeader": show_opacity_in_header,
            "fillRectangle": fill_rectangle,
        }
        self._options_initialized = False

    def update_project_meta(self, project_meta: ProjectMeta):
        self._project_meta = project_meta.clone()

    def set_item(self, image_url, ann: Union[Annotation, dict] = None):
        self._image_url = image_url
        self.set_annotation(ann)

    def set_annotation(self, ann: Union[Annotation, dict] = None):
        if ann is not None:
            if type(ann) is dict:
                res_ann = Annotation.from_json(ann, self._project_meta)
            else:
                res_ann = ann.clone()
        else:
            res_ann = None
        self._ann = res_ann

    def _get_item_annotation(self):
        figures = [label.to_json() for label in self._ann.labels] if self._ann is not None else None
        return {
            "url": self._image_url,
            "figures": figures,
        }

    def to_json(self):
        if self._image_url is None:
            return None
        return {
            "content": {
                "projectMeta": self._project_meta.to_json(),
                "annotations": {
                    "ann1": self._get_item_annotation(),
                },
                "layout": [["ann1"]]
            },
            "options": {
                **self._options
            }
        }

    def update(self, output=False):
        if self._image_url is None:
            raise ValueError("Image (URL) is not defined")
        content = self.to_json()

        if self._options_initialized is False:
            self._api.task.set_field(self._task_id, self._v_model, content)
            self._options_initialized = True
        else:
            self._api.task.set_field(self._task_id, f"{self._v_model}.content", content["content"])

        if output:
            return content
