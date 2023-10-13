import typing
import os
from os import PathLike
import jinja2
from fastapi.templating import Jinja2Templates as _fastapi_Jinja2Templates
from starlette.templating import _TemplateResponse as _TemplateResponse
from starlette.background import BackgroundTask
from supervisely.app.singleton import Singleton
from supervisely.app.widgets_context import JinjaWidgets

# https://github.com/supervisely/js-bundle
js_bundle_version = "2.1.64"

# https://github.com/supervisely-ecosystem/supervisely-app-frontend-js
js_frontend_version = "0.0.48"


class Jinja2Templates(_fastapi_Jinja2Templates, metaclass=Singleton):
    def __init__(self, directory: typing.Union[str, PathLike] = "templates") -> None:
        super().__init__(directory)

    def _create_env(self, directory: typing.Union[str, PathLike]) -> "jinja2.Environment":
        env_fastapi = super()._create_env(directory)
        env_sly = jinja2.Environment(
            loader=env_fastapi.loader,
            autoescape=True,
            variable_start_string="{{{",
            variable_end_string="}}}",
        )
        env_sly.globals["url_for"] = env_fastapi.globals["url_for"]
        return env_sly

    def TemplateResponse(
        self,
        name: str,
        context: dict,
        status_code: int = 200,
        headers: dict = None,
        media_type: str = None,
        background: BackgroundTask = None,
    ) -> _TemplateResponse:
        from supervisely.app.fastapi.subapp import get_name_from_env

        context_with_widgets = {
            **context,
            **JinjaWidgets().context,
            "js_bundle_version": js_bundle_version,
            "js_frontend_version": js_frontend_version,
            "app_name": get_name_from_env(default="Supervisely App"),
        }

        return super().TemplateResponse(
            name, context_with_widgets, status_code, headers, media_type, background
        )
