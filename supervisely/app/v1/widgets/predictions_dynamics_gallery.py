from typing import Union
from supervisely.project.project_meta import ProjectMeta
from supervisely.api.api import Api
from supervisely.annotation.annotation import Annotation
from supervisely.sly_logger import logger
from typing import List
from supervisely._utils import rand_str

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


class PredictionsDynamicsGallery:
    def __init__(self, task_id, api: Api, v_model: str, project_meta: ProjectMeta):
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        if self._v_model.startswith("data.") is False:
            new_v_model = v_model.replace('state.', 'data.', 1)
            raise KeyError(f"Data for this widget has to be stored in data field due to potential performance issues. "
                           f"Please, change value of v_model argument "
                           f"from {v_model} to {new_v_model} manually and also check html template")

        self._project_meta = project_meta.clone()
        self._options = {
            "enableZoom": True,
            "syncViews": True,
            "showPreview": True,
            "selectable": True,
            "opacity": 0.5,
            "showOpacityInHeader": True,
            #"viewHeight": 450,
        }
        self._num_cols = 3

        self._gt_annotations = {}
        self._pred_annotations = {}
        self._order = []

        self._infos = {}
        self._min_time_index = None
        self._max_time_index = None
        self._active_time = None

        self._items_to_upload = {
            "gt": [],
            #"pred": {}
        }

    def has_item(self, name):
        return name in self._gt_annotations

    def create_item(self, name: str, image_url: str, gt_ann: Annotation):
        if name in self._gt_annotations:
            raise KeyError(f"Item with name {name} already exists")
        self._gt_annotations[name] = gt_ann
        self._pred_annotations[name] = {}
        self._infos[name] = {
            "url": image_url,
        }
        self._items_to_upload["gt"].append(name)
        self._order.append(name)

    def _update_indices_range(self, time_index):
        def _get(cur_val, new_value, func):
            return new_value if cur_val is None else func(cur_val, new_value)
        self._min_time_index = _get(self._min_time_index, time_index, min)
        self._max_time_index = _get(self._min_time_index, time_index, max)

    def add_prediction(self, name, time_index: int, pred_ann: Annotation):
        if time_index in self._pred_annotations[name]:
            raise KeyError(f"Prediction for item {name} already exists for time index {time_index}")
        self._pred_annotations[name][time_index] = pred_ann
        self._update_indices_range(time_index)
        #self._items_to_upload["pred"][name] = time_index

    def complete_update(self):
        gallery_json = self.to_json()
        fields = [{"field": self._v_model, "payload": gallery_json}]
        self._api.task.set_fields(self._task_id, fields)

    def partial_update(self):
        active_time = self._get_active_time()
        update_anns = {}
        for name in self._items_to_upload["gt"]:
            update_anns[f"{name}_image"] = self._get_item_annotation(name, None, "image")
            update_anns[f"{name}_gt"] = self._get_item_annotation(name, self._gt_annotations[name], "gt")
            if active_time not in self._pred_annotations[name]:
                raise KeyError(f"Item {name} does not have prediction for time index {active_time}")
            update_anns[f"{name}_pred"] = self._get_item_annotation(name, self._pred_annotations[name][active_time], "pred")

        layout, sync_view = self._construct_layout()

        fields_sync = {"field": f"{self._v_model}.options.syncViewsBindings", "payload": sync_view}
        fields = [
            {"field": f"{self._v_model}.content.annotations", "payload": update_anns, "append": True},
            {"field": f"{self._v_model}.content.layout", "payload": layout},
            fields_sync
        ]
        self._api.task.set_fields(self._task_id, fields)
        self._api.task.set_fields(self._task_id, [fields_sync])

        self._items_to_upload["gt"].clear()
        #self._items_to_upload["pred"].clear()

    def update(self, partial=True):
        if partial is True:
            self.partial_update()
        else:
            self.complete_update()

    def _get_item_annotation(self, name, ann: Annotation, title):
        res = {
            "url": self._infos[name]["url"],
            "figures": [],
            "title": title,
        }
        if ann is not None:
            res["figures"] = [label.to_json() for label in ann.labels]
        return res

    #@TODO: reimplement
    def to_json(self):
        layout, sync_view = self._construct_layout()

        annotations = {}
        for name in self._order:
            annotations[f"{name}_image"] = self._get_item_annotation(name, None, "image")
            annotations[f"{name}_gt"] = self._get_item_annotation(name, self._gt_annotations[name], "gt")
            # @TODO: reimplement
            for ti, pred_ann in self._pred_annotations[name].items():
                annotations[f"{name}_pred"] = self._get_item_annotation(name, pred_ann, "pred")

        return {
            "content": {
                "projectMeta": self._project_meta.to_json(),
                "layout": layout,
                "annotations": annotations,
                "pred_layout": layout[2]
            },
            "options": {
                **self._options,
                "syncViewsBindings": sync_view
            }
        }

    def _get_active_time(self):
        if self._active_time is None:
            return self._max_time_index
        return self._active_time

    def _construct_layout(self):
        layout = [[] for i in range(self._num_cols)]
        sync_view = []
        for name in self._order:
            layout[0].append(f"{name}_image")
            layout[1].append(f"{name}_gt")
            active_time = self._get_active_time()
            if active_time not in self._pred_annotations[name]:
                raise KeyError(f"Item {name} does not have prediction for time index {active_time}")
            layout[2].append(f"{name}_pred")
            sync_view.append([f"{name}_image", f"{name}_gt", f"{name}_pred"])
        return layout, sync_view

    def is_show_last_time(self):
        return self._active_time is None

    def show_last_time_index(self):
        time_index = self._max_time_index
        self._set_time_index(time_index)

    def follow_last_time_index(self):
        self._active_time = None

    def set_time_index(self, time_index):
        self._active_time = time_index
        self._set_time_index(time_index)

    def _set_time_index(self, time_index):
        pred_update = {}
        for name in self._order:
            if time_index not in self._pred_annotations[name]:
                raise KeyError(f"Item {name} does not have prediction for time index {time_index}")
            pred_update[f"{name}_pred"] = self._get_item_annotation(name, self._pred_annotations[name][time_index],
                                                                    "pred")

        layout, sync_view = self._construct_layout()

        fields_sync = {"field": f"{self._v_model}.options.syncViewsBindings", "payload": sync_view}
        fields = [
            {"field": f"{self._v_model}.content.annotations", "payload": pred_update, "append": True},
            {"field": f"{self._v_model}.content.layout", "payload": layout},
            fields_sync
        ]
        self._api.task.set_fields(self._task_id, fields)
        self._api.task.set_fields(self._task_id, [fields_sync])
