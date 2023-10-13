from typing import Union
from supervisely.project.project_meta import ProjectMeta
from supervisely.api.api import Api
from supervisely.annotation.annotation import Annotation


# html example
# <sly-grid-gallery
#         v-if="data.gallery"
#         :content="data.gallery.content"
#         :options="data.gallery.options">
#     <template v-slot:card-footer="{ annotation }">
#         <div class="mt5" style="text-align: center">
#             <el-tag type="primary">{{annotation.title}}</el-tag>
#         </div>
#     </template>
# </sly-grid-gallery>


class CompareGallery:

    def __init__(self, task_id, api: Api, v_model, project_meta: ProjectMeta):
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        self._project_meta = project_meta.clone()

        self._left_title = None
        self._left_image_url = None
        self._left_ann = None

        self._right_title = None
        self._right_image_url = None
        self._right_ann = None

        self._options = {
            "enableZoom": True,
            "syncViews": True,
            "showPreview": False,
            "selectable": False,
            "opacity": 0.5,
            "showOpacityInHeader": True,
            # "viewHeight": 450,
        }
        self._options_initialized = False

    def update_project_meta(self, project_meta: ProjectMeta):
        self._project_meta = project_meta.clone()

    def _set_item(self, name, title, image_url, ann: Union[Annotation, dict] = None):
        setattr(self, f"_{name}_title", title)
        setattr(self, f"_{name}_image_url", image_url)
        res_ann = Annotation((1,1))
        if ann is not None:
            if type(ann) is dict:
                res_ann = Annotation.from_json(ann, self._project_meta)
            else:
                res_ann = ann.clone()
        setattr(self, f"_{name}_ann", res_ann)

    def set_left(self, title, image_url, ann: Union[Annotation, dict] = None):
        self._set_item("left", title, image_url, ann)

    def set_right(self, title, image_url, ann: Union[Annotation, dict] = None):
        self._set_item("right", title, image_url, ann)

    def _get_item_annotation(self, name):
        return {
            "url": getattr(self, f"_{name}_image_url"),
            "figures": [label.to_json() for label in getattr(self, f"_{name}_ann").labels],
            "title": getattr(self, f"_{name}_title"),
        }

    def update(self, options=True):
        if self._left_image_url is None:
            raise ValueError("Left item is not defined")
        if self._right_image_url is None:
            raise ValueError("Right item is not defined")
        gallery_json = self.to_json()
        if options is True or self._options_initialized is False:
            self._api.task.set_field(self._task_id, self._v_model, gallery_json)
            self._options_initialized = True
        else:
            self._api.task.set_field(self._task_id, f"{self._v_model}.content", gallery_json["content"])

    def to_json(self):
        if self._left_image_url is None or self._right_image_url is None:
            return None

        return {
            "content": {
                "projectMeta": self._project_meta.to_json(),
                "layout": [["left"], ["right"]],
                "annotations": {
                    "left": self._get_item_annotation("left"),
                    "right": self._get_item_annotation("right"),
                }
            },
            "options": {
                **self._options,
                "syncViewsBindings": [["left", "right"]]
            }
        }
