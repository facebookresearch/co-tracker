from functools import wraps
from typing import List, Dict, Union, Optional, Callable

import yaml
import supervisely.app.widgets as Widgets
import supervisely.io.env as env
from supervisely.task.progress import Progress
from supervisely import Api
from supervisely.api.file_api import FileApi
from supervisely.sly_logger import logger


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class BaseInferenceGUI:
    @property
    def serve_button(self) -> Widgets.Button:
        # return self._serve_button
        raise NotImplementedError("Have to be implemented in child class")

    @property
    def download_progress(self) -> Widgets.Progress:
        # return self._download_progress
        raise NotImplementedError("Have to be implemented in child class")

    def get_device(self) -> str:
        # return "cpu"
        raise NotImplementedError("Have to be implemented in child class")

    def set_deployed(self) -> None:
        raise NotImplementedError("Have to be implemented in child class")

    def get_ui(self) -> Widgets.Widget:
        # return Widgets.Container(widgets_list)
        raise NotImplementedError("Have to be implemented in child class")


CallbackT = Callable[[BaseInferenceGUI], None]


class InferenceGUI(BaseInferenceGUI):
    def __init__(
        self,
        models: Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]],
        api: Api,
        support_pretrained_models: Optional[bool],
        support_custom_models: Optional[bool],
        add_content_to_pretrained_tab: Optional[Callable] = None,
        add_content_to_custom_tab: Optional[Callable] = None,
        custom_model_link_type: Optional[Literal["file", "folder"]] = "file",
    ):
        if isinstance(models, dict):
            self._support_submodels = True
        else:
            self._support_submodels = False
        self._api = api
        if not support_pretrained_models and not support_custom_models:
            raise ValueError(
                """
                You should provide either pretrained models via get_models() 
                function or allow to use custom models via support_custom_models().
                """
            )
        self._support_custom_models = support_custom_models
        self._support_pretrained_models = support_pretrained_models

        device_values = []
        device_names = []
        try:
            import torch

            if torch.cuda.is_available():
                gpus = torch.cuda.device_count()
                for i in range(gpus):
                    device_values.append(f"cuda:{i}")
                    device_names.append(f"{torch.cuda.get_device_name(i)} (cuda:{i})")
        except:
            pass
        device_values.append("cpu")
        device_names.append("CPU")

        self._device_select = Widgets.SelectString(values=device_values, labels=device_names)
        self._device_field = Widgets.Field(self._device_select, title="Device")
        self._serve_button = Widgets.Button("SERVE")
        self._success_label = Widgets.DoneLabel()
        self._success_label.hide()
        self._download_progress = Widgets.Progress("Downloading model...", hide_on_finish=True)
        self._download_progress.hide()
        self._change_model_button = Widgets.Button(
            "STOP AND CHOOSE ANOTHER MODEL", button_type="danger"
        )
        self._change_model_button.hide()

        self._model_inference_settings_widget = Widgets.Editor(
            readonly=True, restore_default_button=False
        )
        self._model_inference_settings_container = Widgets.Field(
            self._model_inference_settings_widget,
            title="Inference settings",
            description="Model allows user to configure the following parameters on prediction phase",
        )

        self._model_info_widget = Widgets.ModelInfo()
        self._model_info_widget_container = Widgets.Field(
            self._model_info_widget,
            title="Session Info",
            description="Basic information about the deployed model",
        )

        self._model_classes_widget = Widgets.ClassesTable(selectable=False)
        self._model_classes_plug = Widgets.Text("No classes provided")
        self._model_classes_widget_container = Widgets.Field(
            content=Widgets.Container([self._model_classes_widget, self._model_classes_plug]),
            title="Model classes",
            description="List of classes model predicts",
        )

        self._model_full_info = Widgets.Container(
            [
                self._model_info_widget_container,
                self._model_inference_settings_container,
                self._model_classes_widget_container,
            ]
        )
        self._model_full_info.hide()
        self._before_deploy_msg = Widgets.Text("Deploy model to see the information.")

        self._model_full_info_card = Widgets.Card(
            title="Full model info",
            description="Inference settings, session parameters and model classes",
            collapsable=True,
            content=Widgets.Container(
                [
                    self._model_full_info,
                    self._before_deploy_msg,
                ]
            ),
        )

        self._model_full_info_card.collapse()
        self.get_ui = self.add_model_info_to_get_ui(self._model_full_info_card)

        tabs_titles = []
        tabs_contents = []
        tabs_descriptions = []

        if self._support_pretrained_models:
            if self._support_submodels:
                self._model_select = Widgets.SelectString([], filterable=True)

                @self._model_select.value_changed
                def update_table(selected_model):
                    cols = [
                        model_key
                        for model_key in self._models[selected_model]["checkpoints"][0].keys()
                    ]
                    rows = [
                        [value for param_name, value in model.items()]
                        for model in self._models[selected_model]["checkpoints"]
                    ]

                    table_subtitles, cols = self._get_table_subtitles(cols)
                    self._models_table.set_data(cols, rows, table_subtitles)

            self._models_table = None
            self._set_pretrained_models(models)

            pretrained_tab_content = []
            if self._support_submodels:
                pretrained_tab_content.append(self._model_select)
            pretrained_tab_content.append(self._models_table)
            # add user widget after the table
            if add_content_to_pretrained_tab is not None:
                widget_to_add = add_content_to_pretrained_tab(self)
            else:
                widget_to_add = None
            if widget_to_add is not None and not support_pretrained_models:
                raise ValueError(
                    "You can provide content to pretrained models tab only If get_models() is not empty list."
                )
            self.add_to_pretrained_tab = widget_to_add
            if self.add_to_pretrained_tab is not None:
                pretrained_tab_content.append(self.add_to_pretrained_tab)
            tabs_titles.append("Pretrained models")
            tabs_contents.append(Widgets.Container(pretrained_tab_content))
            tabs_descriptions.append("Models trained outside Supervisely")

        if self._support_custom_models:
            self._file_thumbnail = Widgets.FileThumbnail()
            team_files_url = f"{env.server_address()}/files/"

            self._team_files_link = Widgets.Button(
                text="Open Team Files",
                button_type="info",
                plain=True,
                icon="zmdi zmdi-folder",
                link=team_files_url,
            )

            file_api = FileApi(self._api)
            self._model_path_input = Widgets.Input(
                placeholder=f"Path to model {custom_model_link_type} in Team Files"
            )

            @self._model_path_input.value_changed
            def change_folder(value):
                file_info = None
                if value != "":
                    file_info = file_api.get_info_by_path(env.team_id(), value)
                self._file_thumbnail.set(file_info)

            # add user widget to top of tab
            if add_content_to_custom_tab is not None:
                widget_to_add = add_content_to_custom_tab(self)
            else:
                widget_to_add = None

            if widget_to_add is not None and not support_custom_models:
                raise ValueError(
                    "You can provide content to custom models tab only If support_custom_models() returned True."
                )

            self._model_path_field = Widgets.Field(
                self._model_path_input,
                title=f"Copy path to model {custom_model_link_type} from Team Files and paste to field below.",
                description="Copy path in Team Files",
            )

            custom_tab_widgets = [
                self._team_files_link,
                self._model_path_field,
                self._file_thumbnail,
            ]
            self.add_to_custom_tab = widget_to_add
            if self.add_to_custom_tab is not None:
                custom_tab_widgets.append(self.add_to_custom_tab)
            custom_tab_content = Widgets.Container(custom_tab_widgets)
            tabs_titles.append("Custom models")
            tabs_contents.append(custom_tab_content)
            tabs_descriptions.append("Models trained in Supervisely and located in Team Files")

        self._tabs = Widgets.RadioTabs(
            titles=tabs_titles,
            contents=tabs_contents,
            descriptions=tabs_descriptions,
        )

        self.on_change_model_callbacks: List[CallbackT] = [InferenceGUI._hide_info_after_change]
        self.on_serve_callbacks: List[CallbackT] = []

        @self.serve_button.click
        def serve_model():
            for cb in self.on_serve_callbacks:
                cb(self)
            self.set_deployed()

        @self._change_model_button.click
        def change_model():
            for cb in self.on_change_model_callbacks:
                cb(self)
            self.change_model()

    def change_model(self):
        self._success_label.text = ""
        self._success_label.hide()
        self._serve_button.show()
        self._device_select.enable()
        self._change_model_button.hide()
        if self._support_pretrained_models:
            self._models_table.enable()
        if self._support_custom_models:
            self._model_path_input.enable()
        Progress("model deployment canceled", 1).iter_done_report()

    def _hide_info_after_change(self):
        self._model_full_info_card.collapse()
        self._model_full_info.hide()
        self._before_deploy_msg.show()

    def _set_pretrained_models(self, models):
        self._models = models
        if not self._support_pretrained_models:
            return
        if self._support_submodels:
            model_papers = []
            model_years = []
            model_links = []
            for _, model_data in models.items():
                for param_name, param_val in model_data.items():
                    if param_name == "paper_from":
                        model_papers.append(param_val)
                    elif param_name == "year":
                        model_years.append(param_val)
                    elif param_name == "config_url":
                        model_links.append(param_val)
            paper_and_year = []
            for paper, year in zip(model_papers, model_years):
                paper_and_year.append(f"{paper} {year}")
            self._model_select.set(
                list(models.keys()),
                right_text=paper_and_year,
                items_links=model_links,
            )
            selected_model = self._model_select.get_value()
            cols = list(models[selected_model]["checkpoints"][0].keys())
            rows = [
                [value for _, value in model.items()]
                for model in models[selected_model]["checkpoints"]
            ]
        else:
            cols = list(models[0].keys())
            rows = [list(model.values()) for model in models]

        table_subtitles, cols = self._get_table_subtitles(cols)
        if self._models_table is None:
            self._models_table = Widgets.RadioTable(cols, rows, subtitles=table_subtitles)
        else:
            self._models_table.set_data(cols, rows, subtitles=table_subtitles)

    def _get_table_subtitles(self, cols):
        # Get subtitles from col's round brackets
        subtitles = {}
        updated_cols = []
        for col in cols:
            guess_brackets = col.split("(")
            if len(guess_brackets) > 1:
                subtitle = guess_brackets[1]
                if ")" not in subtitle:
                    subtitles[col] = None
                    updated_cols.append(col)
                subtitle = subtitle.split(")")[0]
                new_col = guess_brackets[0].strip()
                subtitles[new_col] = subtitle
                updated_cols.append(new_col)
            else:
                subtitles[col] = None
                updated_cols.append(col)
        return subtitles, updated_cols

    def get_device(self) -> str:
        return self._device_select.get_value()

    def get_model_info(self) -> Dict[str, Dict[str, str]]:
        if not self._support_submodels:
            return None
        selected_model = self._model_select.get_value()
        selected_model_info = self._models[selected_model].copy()
        del selected_model_info["checkpoints"]
        return {selected_model: selected_model_info}

    def get_checkpoint_info(self) -> Dict[str, str]:
        model_row = self._models_table.get_selected_row_index()
        if not self._support_submodels:
            checkpoint_info = self._models[model_row]
        else:
            selected_model = self._model_select.get_value()
            checkpoint_info = self._models[selected_model]["checkpoints"][model_row]
        return checkpoint_info

    def get_model_source(self) -> str:
        if not self._support_custom_models:
            return "Pretrained models"
        elif not self._support_pretrained_models:
            return "Custom models"
        return self._tabs.get_active_tab()

    def get_custom_link(self) -> str:
        if not self._support_custom_models:
            return None
        return self._model_path_input.get_value()

    @property
    def serve_button(self) -> Widgets.Button:
        return self._serve_button

    @property
    def download_progress(self) -> Widgets.Progress:
        return self._download_progress

    def set_deployed(self):
        self._success_label.text = f"Model has been successfully loaded on {self._device_select.get_value().upper()} device"
        self._success_label.show()
        self._serve_button.hide()
        self._device_select.disable()
        self._change_model_button.show()
        if self._support_pretrained_models:
            self._models_table.disable()
        if self._support_custom_models:
            self._model_path_input.disable()
        Progress("Model deployed", 1).iter_done_report()

    def show_deployed_model_info(self, inference):
        self.set_inference_settings(inference)
        self.set_project_meta(inference)
        self.set_model_info(inference)
        self._before_deploy_msg.hide()
        self._model_full_info.show()
        self._model_full_info_card.uncollapse()

    def set_inference_settings(self, inference):
        if len(inference.custom_inference_settings_dict.keys()) == 0:
            inference_settings_str = "# inference settings dict is empty"
        else:
            inference_settings_str = yaml.dump(inference.custom_inference_settings_dict)
        self._model_inference_settings_widget.set_text(inference_settings_str, "yaml")
        self._model_inference_settings_widget.show()

    def set_project_meta(self, inference):
        if self._get_classes_from_inference(inference) is None:
            logger.warn("Skip loading project meta.")
            self._model_classes_widget.hide()
            self._model_classes_plug.show()
            return

        # inference.update_model_meta()
        # collapse all info
        # self._model_info_collapse.set_active_panel([])
        self._model_classes_widget.set_project_meta(inference.model_meta)
        self._model_classes_plug.hide()
        self._model_classes_widget.show()

    def set_model_info(self, inference):
        info = inference.get_human_readable_info(replace_none_with="Not provided")
        self._model_info_widget.set_model_info(inference.task_id, info)

    # def create_handler_on_model_changes(
    #     self,
    #     inference,
    #     model_table: Optional[Widgets.RadioTable] = None,
    # ):
    #     self._model_full_info.show()
    #     self._before_deploy_msg.hide()
    #     self.show_deployed_model_info(inference)
    #     if model_table is None:
    #         model_table = self._models_table

    #     if isinstance(model_table, Widgets.RadioTable):

    #         @model_table.value_changed
    #         def upd_models_info_if_possible(new_model):
    #             logger.debug(f"Model changed: {new_model}")
    #             self.show_deployed_model_info(inference)

    #     else:
    #         logger.warn("Failed to create handler for models table")

    def _get_classes_from_inference(self, inference) -> Optional[List[str]]:
        classes = None
        try:
            classes = inference.get_classes()
        except NotImplementedError:
            logger.warn(f"get_classes() function not implemented for {type(inference)} object.")
        except AttributeError:
            logger.warn("Probably, get_classes() function not working without model deploy.")
        except Exception as exc:
            logger.warn("Skip getting classes info due to exception")
            logger.exception(exc)

        if classes is None or len(classes) == 0:
            logger.warn(f"get_classes() function return {classes}; skip classes processing.")
            return None
        return classes

    def get_ui(self) -> Widgets.Widget:
        return Widgets.Container(
            [
                self._tabs,
                self._device_field,
                self._download_progress,
                self._success_label,
                self._serve_button,
                self._change_model_button,
            ],
            gap=3,
        )

    def add_model_info_to_get_ui(
        self,
        model_info_widget: Widgets.Widget,
    ) -> Callable:
        def decorator(get_ui):
            @wraps(get_ui)
            def wrapper(*args, **kwargs):
                ui = get_ui(*args, **kwargs)
                ui_with_info = Widgets.Container([ui, model_info_widget])
                return ui_with_info

            return wrapper

        return decorator(self.get_ui)
