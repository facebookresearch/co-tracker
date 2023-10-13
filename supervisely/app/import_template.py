from os import listdir
from os.path import basename, join, isdir
from typing import List, Literal, Optional, Union

import supervisely.io.env as env
from supervisely._utils import is_development, is_production
from supervisely.api.api import Api
from supervisely.app import DialogWindowError, get_data_dir
from supervisely.app.fastapi.subapp import Application
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    DatasetThumbnail,
    Field,
    FileStorageUpload,
    FileThumbnail,
    FolderThumbnail,
    Input,
    ProjectThumbnail,
    RadioTabs,
    SelectDataset,
    SelectProject,
    SelectWorkspace,
    SlyTqdm,
    Stepper,
    TeamFilesSelector,
    Text,
    Widget,
)
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from tqdm import tqdm

PROJECT_TYPES: List[ProjectType] = [
    ProjectType.IMAGES,
    ProjectType.VIDEOS,
    ProjectType.VOLUMES,
    ProjectType.POINT_CLOUDS,
    ProjectType.POINT_CLOUD_EPISODES,
]
DESTINATION_OPTIONS = ["New Project", "Existing Project", "Existing Dataset"]
DATAS_ELECTION_OPTIONS = ["Drag & Drop", "TeamFiles Selector"]
DATA_TYPES = ["folder", "file"]


class Import(Application):
    class Context:
        def __init__(
            self,
            team_id: int,
            workspace_id: int,
            project_id: int,
            dataset_id: int,
            path: str,
            project_name: str,
            progress: SlyTqdm,
            is_on_agent: bool = False,
        ):
            self._team_id = team_id
            if self._team_id is None:
                raise ValueError(f"Team ID is not specified: {self._team_id}")
            if type(self._team_id) is not int:
                raise ValueError(f"Team ID must be 'int': {self._team_id}")

            self._workspace_id = workspace_id
            if self._workspace_id is None:
                raise ValueError(f"Workspace ID is not specified: {self._workspace_id}")
            if type(self._workspace_id) is not int:
                raise ValueError(f"Workspace ID must be 'int': {self._workspace_id}")

            self._project_id = project_id
            if self._project_id is not None and type(self._project_id) is not int:
                raise ValueError(f"Project ID must be 'int': {self._project_id}")

            self._dataset_id = dataset_id
            if self._dataset_id is not None and type(self._dataset_id) is not int:
                raise ValueError(f"Dataset ID must be 'int': {self._dataset_id}")

            self._path = path
            if self._path is None:
                raise ValueError(f"Remote path is not specified: {self._path}")
            if type(self._path) is not str:
                raise ValueError(f"Remote path must be 'str': {self._path}")

            self._project_name = project_name
            if type(self._project_name) is not str:
                raise ValueError(f"Project name must be 'str': {self._project_name}")

            self._progress = progress

            self._is_on_agent = is_on_agent
            if type(self._is_on_agent) is not bool:
                raise ValueError(f"Flag 'is_on_agent' must be 'bool': {self._is_on_agent}")

        def __str__(self):
            return (
                f"Team ID: {self._team_id}\n"
                f"Workspace ID: {self._workspace_id}\n"
                f"Project ID: {self._project_id}\n"
                f"Dataset ID: {self._dataset_id}\n"
                f"Path: {self._path}\n"
                f"Project name: {self._project_name}\n"
                f"Is on agent: {self._is_on_agent}\n"
            )

        @property
        def team_id(self) -> int:
            return self._team_id

        @property
        def workspace_id(self) -> int:
            return self._workspace_id

        @property
        def project_id(self) -> int:
            return self._project_id

        @property
        def dataset_id(self) -> int:
            return self._dataset_id

        @property
        def path(self) -> str:
            return self._path

        @property
        def project_name(self) -> str:
            return self._project_name

        @property
        def progress(self) -> SlyTqdm or tqdm:
            return self._progress

        @property
        def is_on_agent(self) -> bool:
            return self._is_on_agent

    def __init__(
        self,
        allowed_project_types: List[ProjectType] = None,
        allowed_destination_options: List[
            Literal["New Project", "Existing Project", "Existing Dataset"]
        ] = None,
        allowed_data_type: Literal["folder", "file"] = None,
    ):
        if allowed_project_types is None:
            self._allowed_project_types = PROJECT_TYPES
        else:
            for option in allowed_project_types:
                if option not in PROJECT_TYPES:
                    raise ValueError(f"Project type must be one of {PROJECT_TYPES}: {option}")
            if len(allowed_project_types) > 5:
                raise ValueError(
                    f"Maximum 5 project types can be specified: {allowed_project_types}"
                )
            self._allowed_project_types: List[ProjectType] = allowed_project_types

        if allowed_destination_options is None:
            self._allowed_destination_options = DESTINATION_OPTIONS
        else:
            for option in allowed_destination_options:
                if option not in DESTINATION_OPTIONS:
                    raise ValueError(f"Destination must be one of {DESTINATION_OPTIONS}: {option}")
            if len(allowed_destination_options) > 3:
                raise ValueError(
                    f"Maximum 3 destinations can be specified: {allowed_destination_options}"
                )
            self._allowed_destination_options: Literal[
                "New Project", "Existing Project", "Existing Dataset"
            ] = allowed_destination_options

        if allowed_data_type is None:
            self._allowed_data_type = None
        else:
            if allowed_data_type not in DATA_TYPES:
                raise ValueError(
                    f"Data type must be one of {DATA_TYPES} or 'None' to use both: {allowed_data_type}"
                )

            self._allowed_data_type = allowed_data_type

        self.__layout = None
        api = Api.from_env()

        self.__step_one_btn = Button("Next", icon="zmdi zmdi-check")

        self.__step_one_text = Text()
        self.__step_one_text.hide()

        self.__step_two_btn = Button("Next", icon="zmdi zmdi-check")
        self.__step_two_btn.disable()

        self.__step_three_btn = Button("Next", icon="zmdi zmdi-check")
        self.__step_three_btn.disable()

        self.__step_three_text = Text()
        self.__step_three_text.hide()

        self.__start_button = Button("Start Import")
        self.__start_button.disable()

        self.__output_text = Text()
        self.__output_text.hide()

        self.__output_project_tabs = None
        self.__input_data_tabs = None
        # self.__step_four_btn = Button("Next")

        ####################
        # PROJECT SELECTOR #
        ####################

        self.__input_mode = None

        # Launch from Dataset
        if (
            env.project_id(raise_not_found=False) is not None
            and env.dataset_id(raise_not_found=False) is not None
        ):
            project_info = api.project.get_info_by_id(env.project_id())
            dataset_info = api.dataset.get_info_by_id(env.dataset_id())
            self.__input_thumbnail = DatasetThumbnail(
                project_info=project_info, dataset_info=dataset_info
            )
            self.__project_selector_card_widgets = Container(widgets=[self.__input_thumbnail])
            self.__input_mode = "dataset"

        # Launch from Project
        elif env.project_id(raise_not_found=False) is not None:
            project_info = api.project.get_info_by_id(env.project_id())
            self.__input_thumbnail = ProjectThumbnail(info=project_info)
            self.__project_selector_card_widgets = Container(widgets=[self.__input_thumbnail])
            self.__input_mode = "project"
        # Launch from Ecosystem
        else:
            self.__output_workspace_selector = SelectWorkspace(
                default_id=env.workspace_id(), team_id=env.team_id()
            )
            self.__output_new_project_name = Input(value="My project")
            self.__output_new_project = Container(
                widgets=[
                    self.__output_workspace_selector,
                    Text("Project name"),
                    self.__output_new_project_name,
                ]
            )
            self.__output_project_selector = SelectProject(
                allowed_types=self._allowed_project_types
            )
            self.__output_dataset_selector = SelectDataset(
                allowed_project_types=self._allowed_project_types
            )
            destination_widgets_map = {
                "New Project": self.__output_new_project,
                "Existing Project": self.__output_project_selector,
                "Existing Dataset": self.__output_dataset_selector,
            }

            if len(self._allowed_destination_options) > 1:
                self.__output_project_tabs = RadioTabs(
                    titles=self._allowed_destination_options,
                    contents=[
                        destination_widgets_map[option]
                        for option in self._allowed_destination_options
                    ],
                )
                self.__project_selector_card_widgets = Container(
                    widgets=[self.__output_project_tabs, self.__step_three_text]
                )
            else:
                self.__project_selector_card_widgets = Container(
                    widgets=[
                        destination_widgets_map[self._allowed_destination_options[0]],
                        self.__step_three_text,
                    ]
                )
            self.__input_mode = "ecosystem"

        ####################
        #  FILES SELECTOR  #
        ####################

        # Launch from File
        if is_production():
            if env.file(raise_not_found=False) is not None:
                file_info = api.file.get_info_by_path(team_id=env.team_id(), remote_path=env.file())
                self.__input_thumbnail = FileThumbnail(info=file_info)
                self.__files_selector_card_widgets = Container(widgets=[self.__input_thumbnail])
                if self.__input_mode is None or self.__input_mode == "ecosystem":
                    self.__input_mode = "file"
                self.__step_two_btn.enable()

            # Launch from Folder
            elif env.folder(raise_not_found=False) is not None:
                import tempfile

                temp_file = tempfile.NamedTemporaryFile(suffix=".tmp", delete=False)
                temp_file.close()
                file_path = temp_file.name
                remote_path = join(env.folder(), basename(file_path))
                file_info = api.file.upload(team_id=env.team_id(), src=file_path, dst=remote_path)
                self.__input_thumbnail = FolderThumbnail(info=file_info)
                api.file.remove(team_id=env.team_id(), path=remote_path)
                self.__files_selector_card_widgets = Container(widgets=[self.__input_thumbnail])
                if self.__input_mode is None or self.__input_mode == "ecosystem":
                    self.__input_mode = "folder"
                self.__step_two_btn.enable()

            # Launch from Ecosystem
            else:
                self.__input_file_selector = TeamFilesSelector(
                    team_id=env.team_id(),
                    multiple_selection=False,
                    selection_file_type=self._allowed_data_type,
                    max_height=300,
                )

                app_name = env.app_name(raise_not_found=False)
                if app_name is None:
                    app_name = "import-template"
                else:
                    app_name = app_name.lower().replace(" ", "-")

                if env.task_id(raise_not_found=False) is None:
                    self.__storage_upload_path = f"/import/{app_name}/debug/"
                else:
                    self.__storage_upload_path = f"/import/{app_name}/{env.task_id()}/"

                self.__input_drag_n_drop = FileStorageUpload(
                    team_id=env.team_id(), path=self.__storage_upload_path
                )
                self.__input_data_tabs = RadioTabs(
                    titles=["Drag & Drop", "TeamFiles Selector"],
                    contents=[
                        self.__input_drag_n_drop,
                        self.__input_file_selector,
                    ],
                )
                self.__files_selector_card_widgets = Container(
                    widgets=[self.__input_data_tabs, self.__step_one_text]
                )
                if self.__input_mode is None:
                    self.__input_mode = "ecosystem"

        # if is_development():
        else:
            data_dir = get_data_dir()
            self.__input_file_text = Text()
            if not isdir(data_dir):
                self.__input_file_text.set(
                    f"Please create data folder specified in 'local.env' and put data for import there",
                    "warning",
                )
                self.__step_one_btn.disable()

            if len(listdir(data_dir)) == 0:
                self.__input_file_text.set(
                    f"Please put the data in the data folder: {data_dir}", "warning"
                )
                self.__step_one_btn.disable()

            else:
                self.__input_file_text.set(f"Data folder: {data_dir}", "success")

            self.__files_selector_card_widgets = Container(widgets=[self.__input_file_text])

        ############
        # SETTINGS #
        #############

        self.__remove_source_files_checkbox = Checkbox(
            "Remove source files after import", checked=True
        )

        if is_development():
            self.__remove_source_files_checkbox.uncheck()
            self.__remove_source_files_checkbox.disable()

        self.__settings_card_widgets = Container(widgets=[self.__remove_source_files_checkbox])
        custom_settings_container = self.add_custom_settings()
        if custom_settings_container is not None:
            if not isinstance(custom_settings_container, Widget):
                raise ValueError("Custom settings must return a Widget")
            if type(custom_settings_container) is not Container:
                custom_settings_container = Container(widgets=[custom_settings_container])
            if isinstance(custom_settings_container, Widget):
                custom_field = Field(content=custom_settings_container, title="Custom Settings")
                self.__settings_card_widgets._widgets.append(custom_field)

        self.__settings_card = Card(
            title="Import Settings",
            description="Configure import settings",
            content=self.__settings_card_widgets,
        )

        ##########
        # OUTPUT #
        ##########

        self.__output_project_thumbnail = ProjectThumbnail()
        self.__output_project_thumbnail.hide()
        self.__import_progress = SlyTqdm()
        self.__output_card_container = Container(
            widgets=[
                self.__output_project_thumbnail,
                self.__output_text,
                self.__start_button,
                self.__import_progress,
            ]
        )
        self.__output_card = Card(
            title="Output",
            description="Link to access result project will appear below after successful import",
            content=self.__output_card_container,
        )

        ##########
        # LAYOUT #
        ##########
        if is_production():
            input_files_card_desc = (
                "Select data to import. You can select data from Team Files or use Drag & Drop"
            )
            local_debug_msg = ""
        else:
            input_files_card_desc = "Application was launched in local debug mode. Data folder is specified in local.env file"
            local_debug_msg = " in local debug mode"

        if self.__input_mode == "project":
            self.__input_project_card = Card(
                title="Destination Project",
                description=f"Application was launched from the context menu of the project{local_debug_msg}",
                content=self.__project_selector_card_widgets,
            )
            self.__input_files_card = Card(
                title="Select Data",
                description=f"{input_files_card_desc}",
                content=self.__files_selector_card_widgets,
            )

            self.__input_project_container = Container(widgets=[self.__input_project_card])
            self.__files_selector_container = Container(
                widgets=[self.__input_files_card, self.__step_one_btn]
            )
            self.__settings_container = Container(
                widgets=[self.__settings_card, self.__step_two_btn]
            )

            self.__layout = Stepper(
                widgets=[
                    self.__input_project_container,
                    self.__files_selector_container,
                    self.__settings_container,
                    self.__output_card,
                ],
                active_step=2,
            )

            self.__settings_card.lock(message="Select data to import to unlock this card")
            self.__output_card.lock(message="Configure import settings to unlock this card")

        elif self.__input_mode == "dataset":
            self.__input_project_card = Card(
                title="Destination Dataset",
                description=f"Application was launched from the context menu of the dataset{local_debug_msg}",
                content=self.__project_selector_card_widgets,
            )
            self.__input_files_card = Card(
                title="Select Data",
                description=f"{input_files_card_desc}",
                content=self.__files_selector_card_widgets,
            )

            self.__input_project_container = Container(widgets=[self.__input_project_card])
            self.__files_selector_container = Container(
                widgets=[self.__input_files_card, self.__step_one_btn]
            )
            self.__settings_container = Container(
                widgets=[self.__settings_card, self.__step_two_btn]
            )

            self.__layout = Stepper(
                widgets=[
                    self.__input_project_container,
                    self.__files_selector_container,
                    self.__settings_container,
                    self.__output_card,
                ],
                active_step=2,
            )
            self.__settings_card.lock(message="Select data to import to unlock this card")
            self.__output_card.lock(message="Configure import settings to unlock this card")

        elif self.__input_mode == "file":
            self.__input_project_card = Card(
                title="Destination Project",
                description="Select where do you want to import your data",
                content=self.__project_selector_card_widgets,
            )
            self.__input_files_card = Card(
                title="Input File",
                description=f"Application was launched from the context menu of the file{local_debug_msg}",
                content=self.__files_selector_card_widgets,
            )

            self.__files_selector_container = Container(widgets=[self.__input_files_card])

            self.__settings_container = Container(
                widgets=[self.__settings_card, self.__step_two_btn]
            )
            self.__input_project_container = Container(
                widgets=[self.__input_project_card, self.__step_three_btn]
            )

            self.__layout = Stepper(
                widgets=[
                    self.__files_selector_container,
                    self.__settings_container,
                    self.__input_project_container,
                    self.__output_card,
                ],
                active_step=2,
            )
            self.__input_project_card.lock(message="Configure import settings to unlock this card")
            self.__output_card.lock(message="Select destination project to unlock this card")

        elif self.__input_mode == "folder":
            self.__input_project_card = Card(
                title="Destination Project",
                description="Select where do you want to import your data",
                content=self.__project_selector_card_widgets,
            )
            self.__input_files_card = Card(
                title="Input Folder",
                description=f"Application was launched from the context menu of the folder{local_debug_msg}",
                content=self.__files_selector_card_widgets,
            )

            self.__files_selector_container = Container(widgets=[self.__input_files_card])

            self.__settings_container = Container(
                widgets=[self.__settings_card, self.__step_two_btn]
            )
            self.__input_project_container = Container(
                widgets=[self.__input_project_card, self.__step_three_btn]
            )

            self.__layout = Stepper(
                widgets=[
                    self.__files_selector_container,
                    self.__settings_container,
                    self.__input_project_container,
                    self.__output_card,
                ],
                active_step=2,
            )
            self.__input_project_card.lock(message="Configure import settings to unlock this card")
            self.__output_card.lock(message="Select destination project to unlock this card")

        else:
            self.__input_project_card = Card(
                title="Destination Project",
                description="Select where do you want to import your data",
                content=self.__project_selector_card_widgets,
            )
            self.__input_files_card = Card(
                title="Select Data",
                description=f"{input_files_card_desc}",
                content=self.__files_selector_card_widgets,
            )

            self.__files_selector_container = Container(
                widgets=[self.__input_files_card, self.__step_one_btn]
            )
            self.__settings_container = Container(
                widgets=[self.__settings_card, self.__step_two_btn]
            )
            self.__input_project_container = Container(
                widgets=[self.__input_project_card, self.__step_three_btn]
            )

            self.__layout = Stepper(
                widgets=[
                    self.__files_selector_container,
                    self.__settings_container,
                    self.__input_project_container,
                    self.__output_card,
                ],
                active_step=1,
            )
            self.__settings_card.lock("Select data to import to unlock this card")
            self.__input_project_card.lock(message="Configure import settings to unlock this card")
            self.__output_card.lock(message="Select destination project to unlock this card")
        super().__init__(layout=self.__layout, hot_reload=False)

    def __generate_output_text(self, api: Api):
        team_id, workspace_id, project_id, dataset_id, project_name = self.__get_destination(api)
        if is_production():
            path = self.__get_remote_path()
        else:
            path = get_data_dir()
        team_name = api.team.get_info_by_id(team_id).name
        workspace_name = api.workspace.get_info_by_id(workspace_id).name

        dst_status = "new"
        if project_id is not None:
            project_name = api.project.get_info_by_id(project_id).name
            dst_status = "existing"

        if self.__remove_source_files_checkbox.is_checked():
            remove_source_message = "Temporary data will be removed after a successful import in order to save disk space."
        else:
            remove_source_message = (
                "Data won't be deleted and will remain accessible after a successful import."
            )

        tf_msg = f"<b>You can access data in Team Files by following path:</b> {path}<br>"
        local_m_msg = f"<b>You can access data on local machine by following path:</b> {path}<br>"
        if project_id is not None and dataset_id is not None:
            dataset_name = api.dataset.get_info_by_id(dataset_id).name
            ds_msg = (
                f"<b>Data will be imported to {dst_status} dataset:</b> {dataset_name} (team: {team_name}, workspace: {workspace_name}, project: {project_name})<br>"
                f"<b>{remove_source_message}</b>"
            )

            if is_production():
                output_text = f"{tf_msg}" f"{ds_msg}"
            else:
                output_text = f"{local_m_msg}" f"{ds_msg}"
        else:
            proj_msg = (
                f"<b> Data will be imported to {dst_status} project:</b> {project_name} (team: {team_name}, workspace: {workspace_name})<br>"
                f"<b>{remove_source_message}</b>"
            )

            if is_production():
                output_text = f"{tf_msg}" f"{proj_msg}"
            else:
                output_text = f"{local_m_msg}" f"{proj_msg}"

        return output_text

    def __get_destination(self, api: Api):
        team_id = env.team_id()
        workspace_id = env.workspace_id()
        project_id = None
        dataset_id = None
        project_name = None
        if is_production():
            if self.__input_mode == "project":
                project_id = env.project_id()
            elif self.__input_mode == "dataset":
                project_id = env.project_id()
                dataset_id = env.dataset_id()
            elif self.__input_mode in ["ecosystem", "file", "folder"]:
                project_mode = self.__output_project_tabs.get_active_tab()
                if project_mode == "New Project":
                    team_id = self.__output_workspace_selector._team_selector.get_selected_id()
                    workspace_id = self.__output_workspace_selector.get_selected_id()
                    project_name = self.__output_new_project_name.get_value()
                elif project_mode == "Existing Project":
                    team_id = (
                        self.__output_project_selector._ws_selector._team_selector.get_selected_id()
                    )
                    workspace_id = self.__output_project_selector._ws_selector.get_selected_id()
                    project_id = self.__output_project_selector.get_selected_id()
                    dataset_id = None
                elif project_mode == "Existing Dataset":
                    team_id = (
                        self.__output_dataset_selector._project_selector._ws_selector._team_selector.get_selected_id()
                    )
                    workspace_id = (
                        self.__output_dataset_selector._project_selector._ws_selector.get_selected_id()
                    )
                    project_id = self.__output_dataset_selector._project_selector.get_selected_id()
                    dataset_id = self.__output_dataset_selector.get_selected_id()

        if project_name is None and project_id is not None:
            project_name = api.project.get_info_by_id(project_id).name

        return team_id, workspace_id, project_id, dataset_id, project_name

    def __get_remote_path(self):
        if self.__input_mode == "file":
            path = env.file()
        elif self.__input_mode == "folder":
            path = env.folder()
        elif self.__input_mode in ["ecosystem", "project", "dataset"]:
            data_mode = self.__input_data_tabs.get_active_tab()
            if data_mode == "TeamFiles Selector":
                path = self.__input_file_selector.get_selected_paths()[0]
            elif data_mode == "Drag & Drop":
                path = self.__storage_upload_path

        return path

    def __get_context(self, api: Api) -> Context:
        team_id = env.team_id()
        workspace_id = env.workspace_id()
        project_id = None
        dataset_id = None
        path = None
        project_name = ""
        progress = self.__import_progress
        is_on_agent = False

        if is_production():
            team_id, workspace_id, project_id, dataset_id, project_name = self.__get_destination(
                api
            )

            data_dir = get_data_dir()
            if self.__input_mode == "file":
                path = env.file()
                local_save_path = join(data_dir, basename(path.rstrip("/")))
                api.file.download(team_id=team_id, remote_path=path, local_save_path=path)
                path = local_save_path
            elif self.__input_mode == "folder":
                path = env.folder()
                local_save_path = join(data_dir, basename(path.rstrip("/")))
                api.file.download_directory(
                    team_id=team_id, remote_path=path, local_save_path=local_save_path
                )
                path = local_save_path
            elif self.__input_mode in ["ecosystem", "project", "dataset"]:
                data_mode = self.__input_data_tabs.get_active_tab()
                if data_mode == "TeamFiles Selector":
                    paths = self.__input_file_selector.get_selected_paths()

                    data_path = self.__input_file_selector.get_selected_paths()[0]
                    data_type = self.__input_file_selector.get_selected_items()[0]["type"]

                    path = join(data_dir, basename(data_path.rstrip("/")))
                    if data_type == "file":
                        api.file.download(
                            team_id=team_id, remote_path=data_path, local_save_path=path
                        )
                    else:
                        api.file.download_directory(
                            team_id=team_id, remote_path=data_path, local_save_path=path
                        )
                elif data_mode == "Drag & Drop":
                    paths = self.__input_drag_n_drop.get_uploaded_paths()
                    for path in paths:
                        local_save_path = join(
                            data_dir, path.replace(self.__storage_upload_path, "")
                        )
                        api.file.download(
                            team_id=team_id, remote_path=path, local_save_path=local_save_path
                        )
                    path = data_dir

            return Import.Context(
                team_id=team_id,
                workspace_id=workspace_id,
                project_id=project_id,
                dataset_id=dataset_id,
                path=get_data_dir(),
                project_name=project_name,
                progress=progress,
                is_on_agent=is_on_agent,
            )

        # if is_development():
        else:
            data_dir = get_data_dir()

            if not isdir(data_dir):
                raise ValueError(
                    "Please create data folder specified in 'local.env' and put data for import there"
                )

            if len(listdir(data_dir)) == 0:
                raise ValueError(f"Please put the data in the data folder: {data_dir}")

            # file = env.file(raise_not_found=False)
            # folder = env.folder(raise_not_found=False)

            # if file is not None and folder is not None:
            #     raise KeyError(
            #         "Both FILE and FOLDER envs are defined, but only one is allowed for the import"
            #     )

            # if file is None and folder is None:
            #     raise KeyError(
            #         "One of the environment variables has to be defined for the import app: FILE or FOLDER"
            #     )

            # path = folder
            # if file is not None:
            #     path = file

            is_on_agent = False
            # if path is not None:
            #     is_on_agent = api.file.is_on_agent(path)

            project_id = env.project_id(raise_not_found=False)
            dataset_id = env.dataset_id(raise_not_found=False)
            if project_id is not None:
                project = api.project.get_info_by_id(id=project_id)
                if project is None:
                    raise ValueError(
                        f"Project with ID: '{project_id}' is not found or either archived"
                    )
                project_id = project.id
                logger.info(f"Importing to existing Project: id={project.id}, name={project.name}")

            if dataset_id is not None:
                dataset = api.dataset.get_info_by_id(id=dataset_id)
                if dataset is None:
                    raise ValueError(
                        f"Dataset with ID: '{dataset_id}' is not found or either archived"
                    )
                if project is not None and dataset not in api.dataset.get_list(
                    project_id=project.id
                ):
                    raise ValueError(
                        f"Dataset {dataset.name}(ID {dataset.id}) "
                        f"does not belong to project {project.name} (ID {project.id})."
                    )
                dataset_id = dataset.id
                logger.info(f"Importing to existing Dataset: id={dataset.id}, name={dataset.name}")

            return Import.Context(
                team_id=team_id,
                workspace_id=workspace_id,
                project_id=project_id,
                dataset_id=dataset_id,
                path=get_data_dir(),
                project_name=project_name,
                progress=progress,
                is_on_agent=is_on_agent,
            )

    def add_custom_settings(self) -> List[Widget]:
        # implement your own method for import settings
        return None

    def process(self, context: Context) -> Optional[Union[int, None]]:
        # implement your own method for import
        raise NotImplementedError()

    def run(self):
        api = Api.from_env()

        # if is_development():
        #     context = self.__get_context(api)
        #     project_id = self.process(context)
        #     info = api.project.get_info_by_id(project_id)
        #     logger.info(f"Result project: id={info.id}, name={info.name}")

        # if is_production() is True:
        if self.__input_data_tabs is not None:

            @self.__input_data_tabs.value_changed
            def hide_message_files(value):
                self.__step_one_text.hide()

        if self.__output_project_tabs is not None:

            @self.__output_project_tabs.value_changed
            def hide_message_project(value):
                self.__step_three_text.hide()

        @self.__step_one_btn.click
        def finish_step_one():
            if self.__input_data_tabs is not None:
                if self.__input_data_tabs.get_active_tab() == "TeamFiles Selector":
                    if len(self.__input_file_selector.get_selected_items()) == 0:
                        self.__step_one_text.show()
                        self.__step_one_text.set(text="Select a data to import", status="error")
                        return
                elif self.__input_data_tabs.get_active_tab() == "Drag & Drop":
                    if len(self.__input_drag_n_drop.get_uploaded_paths()) == 0:
                        self.__step_one_text.show()
                        self.__step_one_text.set(text="Drag & Drop data to import", status="error")
                        return

            if self.__step_one_btn.text == "Next":
                self.__step_one_text.hide()

                if self.__input_mode in ["dataset", "project"]:
                    self.__input_project_card.unlock()
                    self.__input_files_card.lock(
                        message="Press the button to restart from this step"
                    )
                    self.__layout.set_active_step(3)

                elif self.__input_mode in ["file", "folder"]:
                    self.__input_project_card.unlock()
                    self.__layout.set_active_step(3)

                elif self.__input_mode == "ecosystem":
                    self.__input_files_card.lock(
                        message="Press the button to restart from this step"
                    )
                    self.__input_project_card.lock()
                    self.__layout.set_active_step(2)

                self.__step_one_btn.text = "Restart"
                self.__step_one_btn.icon = "zmdi zmdi-rotate-left"

                self.__settings_card.unlock()
                self.__step_two_btn.enable()

            elif self.__step_one_btn.text == "Restart":
                self.__step_one_text.hide()

                if self.__input_mode in ["dataset", "project"]:
                    self.__input_project_card.unlock()
                    self.__input_files_card.unlock()
                    self.__output_card.lock(message="Configure import settings to unlock this card")
                    self.__layout.set_active_step(2)
                elif self.__input_mode in ["file", "folder"]:
                    self.__input_project_card.unlock()
                    self.__layout.set_active_step(2)
                    self.__output_card.lock(
                        message="Select destination project to unlock this card"
                    )
                elif self.__input_mode == "ecosystem":
                    self.__input_files_card.unlock()
                    self.__input_project_card.lock(
                        message="Configure import settings to unlock this card"
                    )
                    self.__layout.set_active_step(1)
                    self.__output_card.lock(
                        message="Select destination project to unlock this card"
                    )

                self.__step_two_btn.text = "Next"
                self.__step_two_btn.icon = "zmdi zmdi-check"
                self.__step_three_btn.text = "Next"
                self.__step_three_btn.icon = "zmdi zmdi-check"

                self.__step_one_btn.text = "Next"
                self.__step_one_btn.icon = "zmdi zmdi-check"
                self.__step_two_btn.disable()
                self.__step_three_btn.disable()
                self.__settings_card.lock(message="Select data to import to unlock this card")

            self.__output_text.hide()
            self.__start_button.disable()

        @self.__step_two_btn.click
        def finish_step_two():
            if self.__step_two_btn.text == "Next":
                if self.__input_mode in ["dataset", "project"]:
                    self.__input_files_card.lock()
                    self.__settings_card.lock(message="Press the button to restart from this step")

                    output_text = self.__generate_output_text(api)
                    self.__output_text.set(
                        text=output_text,
                        status="info",
                    )
                    self.__output_text.show()
                    self.__start_button.enable()
                    self.__output_card.unlock()
                    self.__layout.set_active_step(4)

                elif self.__input_mode in ["file", "folder"]:
                    self.__input_project_card.unlock()
                    self.__settings_card.lock(message="Press the button to restart from this step")
                    self.__layout.set_active_step(3)
                elif self.__input_mode == "ecosystem":
                    self.__input_files_card.lock()
                    self.__input_project_card.unlock()
                    self.__settings_card.lock(message="Press the button to restart from this step")
                    self.__layout.set_active_step(3)

                self.__step_two_btn.text = "Restart"
                self.__step_two_btn.icon = "zmdi zmdi-rotate-left"
                self.__step_three_btn.enable()

            elif self.__step_two_btn.text == "Restart":
                if self.__input_mode in ["dataset", "project"]:
                    self.__input_files_card.lock()
                    self.__settings_card.unlock()
                    self.__layout.set_active_step(3)
                elif self.__input_mode in ["file", "folder"]:
                    self.__settings_card.unlock()
                    self.__input_project_card.lock(
                        message="Configure import settings to unlock this card"
                    )
                    self.__layout.set_active_step(2)
                elif self.__input_mode == "ecosystem":
                    self.__input_files_card.lock()
                    self.__settings_card.unlock()
                    self.__input_project_card.lock(
                        message="Configure import settings to unlock this card"
                    )
                    self.__layout.set_active_step(2)

                self.__step_three_btn.text = "Next"
                self.__step_three_btn.icon = "zmdi zmdi-check"
                self.__step_three_btn.disable()
                self.__output_card.lock()

                self.__output_text.hide()
                self.__start_button.disable()
                self.__step_two_btn.text = "Next"
                self.__step_two_btn.icon = "zmdi zmdi-check"

        @self.__step_three_btn.click
        def finish_step_three():
            if (
                self.__output_project_tabs is not None
                and len(self._allowed_destination_options) > 1
            ):
                if self.__output_project_tabs.get_active_tab() == "New Project":
                    if self.__output_new_project_name.get_value() == "":
                        self.__step_three_text.show()
                        self.__step_three_text.set(text="Enter a project name", status="error")
                        return
                elif self.__output_project_tabs.get_active_tab() == "Existing Project":
                    if self.__output_project_selector.get_selected_id() is None:
                        self.__step_three_text.show()
                        self.__step_three_text.set(
                            text="Select a project to import to", status="error"
                        )
                        return
                elif self.__output_project_tabs.get_active_tab() == "Existing Dataset":
                    if (
                        self.__output_dataset_selector.get_selected_id() is None
                        or type(self.__output_dataset_selector.get_selected_id()) is not int
                    ):
                        self.__step_three_text.show()
                        self.__step_three_text.set(
                            text="Select a dataset to import to", status="error"
                        )
                        return

            else:
                if self._allowed_destination_options == ["New Project"]:
                    if self.__output_new_project_name.get_value() == "":
                        self.__step_three_text.show()
                        self.__step_three_text.set(text="Enter a project name", status="error")
                        return
                elif self._allowed_destination_options == ["Existing Project"]:
                    if self.__output_project_selector.get_selected_id() is None:
                        self.__step_three_text.show()
                        self.__step_three_text.set(
                            text="Select a project to import to", status="error"
                        )
                        return
                elif self._allowed_destination_options == ["Existing Dataset"]:
                    if (
                        self.__output_dataset_selector.get_selected_id() is None
                        or len(self.__output_dataset_selector.get_selected_id()) == 0
                    ):
                        self.__step_three_text.show()
                        self.__step_three_text.set(
                            text="Select a dataset to import to", status="error"
                        )
                        return

            if self.__step_three_btn.text == "Next":
                self.__step_three_text.hide()
                if self.__input_mode in ["dataset", "project"]:
                    self.__input_files_card.unlock()
                elif self.__input_mode in ["file", "folder"]:
                    self.__settings_card.lock()
                    self.__input_project_card.lock(
                        message="Press the button to restart from this step"
                    )
                elif self.__input_mode == "ecosystem":
                    self.__settings_card.lock()
                    self.__input_project_card.lock(
                        message="Press the button to restart from this step"
                    )

                output_text = self.__generate_output_text(api)
                self.__output_text.set(
                    text=output_text,
                    status="info",
                )
                self.__output_text.show()

                self.__start_button.enable()
                self.__layout.set_active_step(4)

                self.__step_three_btn.text = "Restart"
                self.__step_three_btn.icon = "zmdi zmdi-rotate-left"
                self.__output_card.unlock()

            elif self.__step_three_btn.text == "Restart":
                self.__step_three_text.hide()
                if self.__input_mode in ["dataset", "project"]:
                    self.__input_files_card.unlock()
                    self.__output_card.lock(message="Configure import settings to unlock this card")
                elif self.__input_mode in ["file", "folder"]:
                    self.__settings_card.lock()
                    self.__input_project_card.unlock()
                    self.__output_card.lock(
                        message="Select destination project to unlock this card"
                    )
                elif self.__input_mode == "ecosystem":
                    self.__input_project_card.unlock()
                    self.__settings_card.lock()
                    self.__output_card.lock(
                        message="Select destination project to unlock this card"
                    )

                self.__layout.set_active_step(3)

                self.__output_text.hide()
                self.__start_button.disable()
                self.__step_three_btn.text = "Next"
                self.__step_three_btn.icon = "zmdi zmdi-check"

        @self.__start_button.click
        def start_import():
            try:
                self.__step_one_btn.disable()
                self.__step_two_btn.disable()
                self.__step_three_btn.disable()
                self.__output_text.hide()
                context = self.__get_context(api)
                self.__import_progress.show()

                project_id = self.process(context)
                if type(project_id) is int:
                    info = api.project.get_info_by_id(project_id)
                    if is_production() and self.__remove_source_files_checkbox.is_checked():
                        try:
                            dd_paths = self.__input_drag_n_drop.get_uploaded_paths()
                            tf_paths = self.__input_file_selector.get_selected_paths()
                            paths = dd_paths + tf_paths
                            for path in paths:
                                api.file.remove(team_id=env.team_id(), path=path)
                            self.__output_text.set(
                                text="Source files have been successfully removed",
                                status="success",
                            )
                            self.__output_text.show()
                        except Exception as e:
                            self.__output_text.set(
                                text=f"Couldn't remove source files<br>Error: {e}",
                                status="error",
                            )
                            self.__output_text.show()

                    logger.info(f"Result project: id={info.id}, name={info.name}")
                    self.__output_project_thumbnail.set(info)
                    self.__output_text.set("Import completed successfully", status="success")

                    self.__output_project_thumbnail.show()
                    self.__output_text.show()
                else:
                    self.__step_one_btn.enable()
                    self.__step_two_btn.enable()
                    self.__step_three_btn.enable()
                    raise DialogWindowError(
                        title="Import error",
                        description=f"Error: '{project_id}' is not a valid project id",
                    )

                self.__start_button.disable()
            except Exception as e:
                self.__import_progress.hide()
                self.__output_project_thumbnail.hide()
                self.__step_one_btn.enable()
                self.__step_two_btn.enable()
                self.__step_three_btn.enable()
                self.__start_button.enable()
                self.__start_button.loading = False
                raise DialogWindowError(title="Import error", description=f"Error: {str(e)}")
