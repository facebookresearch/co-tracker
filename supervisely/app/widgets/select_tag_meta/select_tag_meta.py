from typing import Dict, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.app.widgets.select_sly_utils import _get_int_or_env, _get_int_env
from supervisely.annotation.tag_meta import TagMeta, SUPPORTED_TAG_VALUE_TYPES
from supervisely.project.project_meta import ProjectMeta


class SelectTagMeta(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        default: str = None,
        project_id: int = None,
        project_meta: ProjectMeta = None,
        allowed_types: List[str] = None,
        multiselect: bool = False,
        show_label: bool = True,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
    ):
        self._changes_handled = False
        self._api = Api()
        self._default = default
        self._project_id = project_id
        self._project_info = None
        self._project_meta: ProjectMeta = project_meta
        if project_meta is not None and project_id is not None:
            raise ValueError(
                "You can not provide both project_id and project_meta parameters to SelectTagMeta widget."
            )

        self._allowed_types = allowed_types
        self._multiselect = multiselect
        self._show_label = show_label
        self._size = size

        if allowed_types is not None:
            for value_type in allowed_types:
                if value_type not in SUPPORTED_TAG_VALUE_TYPES:
                    raise ValueError(
                        "value_type = {!r} is unknown, should be one of {}".format(
                            value_type, SUPPORTED_TAG_VALUE_TYPES
                        )
                    )

        if project_meta is None:
            self._project_id = _get_int_or_env(self._project_id, "context.projectId")
        self._tags = None
        self._value = None
        if self._project_id is None and self._project_meta is None:
            dataset_id = _get_int_env("context.datasetId")
            if dataset_id is None:
                raise ValueError(
                    "Argument 'project_id' or environment variables 'context.projectId' or 'context.datasetId' has to be defined"
                )
            dataset_info = self._api.dataset.get_info_by_id(dataset_id, raise_error=True)
            self._project_id = dataset_info.project_id

            self._project_info = self._api.project.get_info_by_id(
                self._project_id, raise_error=True
            )

        elif self._project_meta is not None:
            self._tags = self._project_meta.tag_metas.to_json()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {}
        res["options"] = {
            "showLabel": self._show_label,
            "filterable": True,
            "onlyAvailable": True,
            "multiple": self._multiselect,
        }
        if self._allowed_types is not None:
            res["options"]["availableValueTypes"] = self._allowed_types
        if self._size is not None:
            res["options"]["size"] = self._size
        if self._tags is not None:
            res["tags"] = self._tags
        return res

    def get_json_state(self) -> Dict:
        val = self._default
        if val is not None:
            if len(self._tags) == 0:
                val = None
            elif val not in [t["name"] for t in self._tags]:
                val = self._tags[0]["name"]
        res = {
            "projectId": self._project_id,
            "tags": val,
        }
        if self._multiselect is True:
            res["tags"] = [val]
        return res

    def get_selected_name(self) -> str:
        if self._multiselect is True:
            raise RuntimeError(
                "Tag selector allows multiselect, please use 'get_selected_names' method instead of 'get_selected_name'"
            )
        if StateJson()[self.widget_id]["tags"] == "":
            return None
        return StateJson()[self.widget_id]["tags"]

    def get_selected_names(self) -> List[str]:
        if self._multiselect is False:
            raise RuntimeError(
                "Tag selector does not allow multiselect, please use 'get_selected_name' method instead of 'get_selected_names'"
            )
        if StateJson()[self.widget_id]["tags"] is None:
            return None
        if StateJson()[self.widget_id]["tags"] == []:
            return None
        return StateJson()[self.widget_id]["tags"]

    def get_tag_meta_by_name(self, name: str) -> TagMeta:
        if self._project_id and (
            self._project_meta is None or self._project_meta.tag_metas.get(name) is None
        ):
            self._project_meta = ProjectMeta.from_json(self._api.project.get_meta(self._project_id))
        tag_meta = self._project_meta.tag_metas.get(name)
        if tag_meta is None:
            raise ValueError(f"Tag with name {name} not found in project meta")
        return tag_meta

    def get_selected_item(self) -> TagMeta:
        name = self.get_selected_name()
        if name is None:
            return None
        return self.get_tag_meta_by_name(name)

    def get_selected_items(self) -> List[TagMeta]:
        names = self.get_selected_names()
        if names is None:
            return None
        results = [self.get_tag_meta_by_name(name) for name in names]
        return results

    def set_name(self, name: str):
        if self._multiselect is True:
            self.set_names([name])
        else:
            StateJson()[self.widget_id]["tags"] = name
            StateJson().send_changes()

    def set_names(self, names: List[str]):
        if self._multiselect is False:
            raise RuntimeError(
                "Tag selector does not allow multiselect, please use 'set_name' method instead of 'set_names'"
            )
        StateJson()[self.widget_id]["tags"] = names
        StateJson().send_changes()

    def set_project_meta(self, project_meta: ProjectMeta):
        if self._project_id is not None:
            raise ValueError(
                "You can not provide both project_id and project_meta parameters to SelectTagMeta widget."
            )
        self._project_meta = project_meta
        self._tags = self._project_meta.tag_metas.to_json()
        self.update_data()
        self.update_state()
        StateJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(SelectTagMeta.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        async def _click():
            if self._multiselect is True:
                res = self.get_selected_items()
            else:
                res = self.get_selected_item()
            func(res)

        return _click
