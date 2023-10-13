from typing import Optional, List, Dict, Any

from supervisely.app.widgets import Widget
from supervisely.app.widgets.widget import Disableable
import supervisely as sly
from supervisely.geometry.geometry import Geometry
from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.sly_logger import logger
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.geometry.point_3d import Point3d

type_to_zmdi_icon = {
    sly.AnyGeometry: "zmdi zmdi-shape",
    sly.Rectangle: "zmdi zmdi-crop-din",  # "zmdi zmdi-square-o"
    sly.Polygon: "icons8-polygon",  # "zmdi zmdi-edit"
    sly.Bitmap: "zmdi zmdi-brush",
    sly.Polyline: "zmdi zmdi-gesture",
    sly.Point: "zmdi zmdi-dot-circle-alt",
    sly.Cuboid: "zmdi zmdi-ungroup",  #
    sly.GraphNodes: "zmdi zmdi-grain",
    Cuboid3d: "zmdi zmdi-codepen",
    Pointcloud: "zmdi zmdi-cloud-outline",  #  # "zmdi zmdi-border-clear"
    sly.MultichannelBitmap: "zmdi zmdi-layers",  # "zmdi zmdi-collection-item"
    Point3d: "zmdi zmdi-filter-center-focus",  # "zmdi zmdi-select-all"
}


class ClassesTable(Widget):
    class Routes:
        CLASS_SELECTED = "class_selected_cb"

    def __init__(
        self,
        project_meta: Optional[sly.ProjectMeta] = None,
        project_id: Optional[int] = None,
        project_fs: Optional[sly.Project] = None,
        allowed_types: Optional[List[Geometry]] = None,
        selectable: Optional[bool] = True,
        disabled: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        if project_id is not None and project_fs is not None:
            raise ValueError(
                "You can not provide both project_id and project_fs parameters to Classes Table widget."
            )
        self._table_data = []
        self._columns = []
        self._changes_handled = False
        self._global_checkbox = False
        self._checkboxes = []
        self._selectable = selectable
        self._selection_disabled = disabled
        self._loading = False
        self._allowed_types = allowed_types if allowed_types is not None else []
        if project_id is not None:
            self._api = sly.Api()
        else:
            self._api = None
        self._project_id = project_id
        if project_id is not None:
            if project_meta is not None:
                logger.warn(
                    "Both parameters project_id and project_meta were provided to ClassesTable widget. Project meta classes taken from remote project and project_meta parameter is ignored."
                )
            project_meta = sly.ProjectMeta.from_json(self._api.project.get_meta(project_id))
        self._project_fs = project_fs
        if project_fs is not None:
            if project_meta is not None:
                logger.warn(
                    "Both parameters project_fs and project_meta were provided to ClassesTable widget. Project meta classes taken from project_fs.meta and project_meta parameter is ignored."
                )
            project_meta = project_fs.meta
        if project_meta is not None:
            self._update_meta(project_meta=project_meta)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def value_changed(self, func):
        route_path = self.get_route_path(ClassesTable.Routes.CLASS_SELECTED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_classes()
            func(res)

        return _value_changed

    def _update_meta(self, project_meta: sly.ProjectMeta) -> None:
        columns = ["class", "shape"]
        stats = None
        data_to_show = []
        for obj_class in project_meta.obj_classes:
            if len(self._allowed_types) == 0 or obj_class.geometry_type in self._allowed_types:
                data_to_show.append(obj_class.to_json())

        if self._project_id is not None:
            stats = self._api.project.get_stats(self._project_id)
            project_info = self._api.project.get_info_by_id(self._project_id)
            if project_info.type == str(sly.ProjectType.IMAGES):
                columns.extend(["images count", "labels count"])
            elif project_info.type == str(sly.ProjectType.VIDEOS):
                columns.extend(["videos count", "figures count"])
            elif project_info.type in [
                str(sly.ProjectType.POINT_CLOUDS),
                str(sly.ProjectType.POINT_CLOUD_EPISODES),
            ]:
                columns.extend(["pointclouds count", "figures count"])
            elif project_info.type == str(sly.ProjectType.VOLUMES):
                columns.extend(["volumes count", "figures count"])

            class_items = {}
            for item in stats["images"]["objectClasses"]:
                class_items[item["objectClass"]["name"]] = item["total"]

            class_objects = {}
            for item in stats["objects"]["items"]:
                class_objects[item["objectClass"]["name"]] = item["total"]
            for obj_class in data_to_show:
                obj_class["itemsCount"] = class_items[obj_class["title"]]
                obj_class["objectsCount"] = class_objects[obj_class["title"]]

        elif self._project_fs is not None:
            project_stats = self._project_fs.get_classes_stats()

            if type(self._project_fs) == sly.Project:
                columns.extend(["images count", "labels count"])

            elif type(self._project_fs) == sly.VideoProject:
                columns.extend(["videos count", "objects count", "figures count"])

            elif type(self._project_fs) in [sly.PointcloudProject, sly.PointcloudEpisodeProject]:
                columns.extend(["pointclouds count", "objects count", "figures count"])

            elif type(self._project_fs) == sly.VolumeProject:
                columns.extend(["volumes count", "objects count", "figures count"])

            for obj_class in data_to_show:
                obj_class["itemsCount"] = project_stats["items_count"][obj_class["title"]]
                obj_class["objectsCount"] = project_stats["objects_count"][obj_class["title"]]
                if type(self._project_fs) != sly.Project:
                    obj_class["figuresCount"] = project_stats["figures_count"][obj_class["title"]]

        columns = [col.upper() for col in columns]
        if data_to_show:
            table_data = []
            if self._project_id is not None or self._project_fs is not None:
                data_to_show = sorted(
                    data_to_show, key=lambda line: line["objectsCount"], reverse=True
                )
            for line in data_to_show:
                table_line = []
                icon = type_to_zmdi_icon[sly.AnyGeometry]
                for geo_type, icon_text in type_to_zmdi_icon.items():
                    geo_type: Geometry
                    if geo_type.geometry_name() == line["shape"]:
                        icon = icon_text
                        break
                if line["shape"] == "graph":
                    line["shape"] = "graph (keypoints)"
                table_line.extend(
                    [
                        {
                            "name": "CLASS",
                            "data": line["title"],
                            "color": line["color"],
                        },
                        {"name": "SHAPE", "data": line["shape"], "icon": icon},
                    ]
                )
                if "itemsCount" in line.keys():
                    table_line.append({"name": "ITEMS COUNT", "data": line["itemsCount"]})
                if "objectsCount" in line.keys():
                    table_line.append({"name": "OBJECTS COUNT", "data": line["objectsCount"]})
                if "figuresCount" in line.keys():
                    table_line.append({"name": "FIGURES COUNT", "data": line["figuresCount"]})
                table_data.append(table_line)
            self._table_data = table_data
            self._columns = columns
            self._checkboxes = [False] * len(table_data)
            self._global_checkbox = False
        else:
            self._table_data = []
            self._columns = []
            self._checkboxes = []
            self._global_checkbox = False

    def read_meta(self, project_meta: sly.ProjectMeta) -> None:
        self.loading = True
        self._project_fs = None
        self._project_id = None
        self.clear_selection()
        self._update_meta(project_meta=project_meta)

        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson().send_changes()
        self.loading = False

    def read_project(self, project_fs: sly.Project) -> None:
        self.loading = True
        self._project_fs = project_fs
        self._project_id = None
        self.clear_selection()
        self._update_meta(project_meta=project_fs.meta)

        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson().send_changes()
        self.loading = False

    def read_project_from_id(self, project_id: int) -> None:
        self.loading = True
        self._project_fs = None
        self._project_id = project_id
        if self._api is None:
            self._api = sly.Api()
        project_meta = sly.ProjectMeta.from_json(self._api.project.get_meta(project_id))
        self.clear_selection()
        self._update_meta(project_meta=project_meta)

        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson().send_changes()
        self.loading = False

    def get_json_data(self) -> Dict[str, Any]:
        return {
            "table_data": self._table_data,
            "columns": self._columns,
            "loading": self._loading,
            "disabled": self._selection_disabled,
            "selectable": self._selectable,
        }

    @property
    def allowed_types(self) -> List[Geometry]:
        return self._allowed_types

    @property
    def project_id(self) -> int:
        return self._project_id

    @property
    def project_fs(self) -> int:
        return self._project_fs

    @property
    def loading(self) -> bool:
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    def get_json_state(self) -> Dict[str, Any]:
        return {
            "global_checkbox": self._global_checkbox,
            "checkboxes": self._checkboxes,
        }

    def get_selected_classes(self) -> List[str]:
        classes = []
        for i, line in enumerate(self._table_data):
            if StateJson()[self.widget_id]["checkboxes"][i]:
                for col in line:
                    if col["name"] == "CLASS":
                        classes.append(col["data"])
        return classes

    def clear_selection(self) -> None:
        self._global_checkbox = False
        self._checkboxes = [False] * len(self._table_data)
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson().send_changes()

    def set_project_meta(self, project_meta: sly.ProjectMeta):
        self._update_meta(project_meta)
        self.update_data()
        DataJson().send_changes()

    def select_all(self) -> None:
        self._global_checkbox = True
        self._checkboxes = [True] * len(self._table_data)
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson().send_changes()
