from __future__ import annotations
import os
from pathlib import Path
from bs4 import BeautifulSoup
import re
import time
import uuid
from typing import Union, List
from varname import varname
from jinja2 import Environment
import markupsafe
from supervisely.app.jinja2 import create_env
from supervisely.app.content import DataJson, StateJson
from fastapi import FastAPI
from supervisely.app.fastapi import _MainServer
from supervisely.app.widgets_context import JinjaWidgets
from supervisely._utils import generate_free_name, rand_str
from async_asgi_testclient import TestClient
from supervisely.app.fastapi.utils import run_sync


class Hidable:
    def __init__(self):
        self._hide = False

    def is_hidden(self):
        return self._hide

    def hide(self):
        self._hide = True
        DataJson()[self.widget_id]["hide"] = self._hide
        DataJson().send_changes()

    def show(self):
        self._hide = False
        DataJson()[self.widget_id]["hide"] = self._hide
        DataJson().send_changes()

    def get_json_data(self):
        return {"hide": self._hide}

    def get_json_state(self):
        raise {}

    def _wrap_hide_html(self, widget_id, html):
        soup = BeautifulSoup(html, features="html.parser")
        for item in soup:
            if hasattr(item.__class__, "has_attr") and callable(
                getattr(item.__class__, "has_attr")
            ):
                if item.has_attr("v-if"):
                    item["v-if"] = f'({item["v-if"]}) && data.{widget_id}.hide === false'
                else:
                    item["v-if"] = f"!data.{widget_id}.hide"
        # return f'<div v-if="!data.{widget_id}.hide">{html}</div>'
        return str(soup)


class Disableable:
    def __init__(self):
        self._disabled = False

    def is_disabled(self):
        return self._disabled

    def disable(self):
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def get_json_data(self):
        return {"disabled": self._disabled}

    def get_json_state(self):
        raise {}

    def _wrap_disable_html(self, widget_id, html):
        soup = BeautifulSoup(html, features="html.parser")
        results = soup.find_all(re.compile("^el-"))
        for tag in results:
            if not tag.has_attr("disabled") and not tag.has_attr(":disabled"):
                tag[":disabled"] = f"data.{widget_id}.disabled"
        return str(soup)


class Loading:
    def __init__(self):
        self._loading = False
    
    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    def _wrap_loading_html(self, widget_id, html):
        soup = BeautifulSoup(html, features="html.parser")
        results = soup.find_all(recursive=False)
        for tag in results:
            if tag.has_attr("v-loading") or tag.has_attr(":loading"):
                return html
        for tag in results:
            tag["v-loading"] = f"data.{widget_id}.loading"
        return str(soup)


def generate_id(cls_name=""):
    suffix = rand_str(5)  # uuid.uuid4().hex # uuid.uuid4().hex[10]
    if cls_name == "":
        return "autoId" + suffix
    else:
        return cls_name + "AutoId" + suffix


class Widget(Hidable, Disableable, Loading):
    def __init__(self, widget_id: str = None, file_path: str = __file__):
        super().__init__()
        self._sly_app = _MainServer()
        self.widget_id = widget_id
        self._file_path = file_path
        self._loading = False
        self._disabled = False

        if (
            widget_id is not None
            and JinjaWidgets().auto_widget_id is True
            and ("autoId" in widget_id or "AutoId" in widget_id)
        ):
            # regenerate id with class name at the beggining
            self.widget_id = generate_id(type(self).__name__)

        if widget_id is None:
            if JinjaWidgets().auto_widget_id is True:
                self.widget_id = generate_id(type(self).__name__)
            else:
                try:
                    self.widget_id = varname(frame=2)
                except Exception as e:  # Caller doesn\\\'t assign the result directly to variable(s).
                    try:
                        self.widget_id = varname(frame=3)
                    except Exception as e:  # VarnameRetrievingError('Unable to retrieve the ast node.')
                        self.widget_id = generate_id(type(self).__name__)

        self._register()

    def _register(self):
        # get singletons
        data = DataJson()
        data.raise_for_key(self.widget_id)
        self.update_data()

        state = StateJson()
        state.raise_for_key(self.widget_id)
        self.update_state(state=state)

        JinjaWidgets().context[self.widget_id] = self
        # templates = Jinja2Templates()
        # templates.context_widgets[self.widget_id] = self

    def get_json_data(self):
        raise NotImplementedError()

    def get_json_state(self):
        raise NotImplementedError()

    def update_state(self, state=None):
        serialized_state = self.get_json_state()
        if serialized_state is not None:
            if state is None:
                state = StateJson()
            state.setdefault(self.widget_id, {}).update(serialized_state)

    def update_data(self):
        data = DataJson()

        widget_data = self.get_json_data()
        if widget_data is None:
            widget_data = {}
        hidable_data = super().get_json_data()
        disableable_data = super().get_json_data()

        serialized_data = {**widget_data, **hidable_data, **disableable_data}
        if serialized_data is not None:
            data.setdefault(self.widget_id, {}).update(serialized_data)

    def get_route_path(self, route: str) -> str:
        return f"/{self.widget_id}/{route}"

    def add_route(self, app, route):
        def decorator(f):
            existing_cb = DataJson()[self.widget_id].get("widget_routes", {}).get(route)
            if existing_cb is not None:
                raise Exception(
                    f"Route [{route}] already attached to function with name: {existing_cb}"
                )

            app.add_api_route(f"/{self.widget_id}/{route}", f, methods=["POST"])
            DataJson()[self.widget_id].setdefault("widget_routes", {})[route] = f.__name__

            self.update_data()

        return decorator

    def to_html(self):
        current_dir = Path(self._file_path).parent.absolute()
        jinja2_sly_env: Environment = create_env(current_dir)
        html = jinja2_sly_env.get_template("template.html").render({"widget": self})
        # st = time.time()
        html = self._wrap_loading_html(self.widget_id, html)
        html = self._wrap_disable_html(self.widget_id, html)
        # print("---> Time (_wrap_disable_html): ", time.time() - st, " seconds")
        # st = time.time()
        html = self._wrap_hide_html(self.widget_id, html)
        # print("---> time (_wrap_hide_html): ", time.time() - st, " seconds")
        return markupsafe.Markup(html)

    def __html__(self):
        res = self.to_html()
        return res


class ConditionalWidget(Widget):
    def __init__(self, items: List[ConditionalItem], widget_id: str = None, file_path: str = __file__):
        self._items = items
        super().__init__(widget_id=widget_id, file_path=file_path)

    def get_items(self) -> List[ConditionalItem]:
        res = []
        if self._items is not None:
            res.extend(self._items)
        return res


class ConditionalItem:
    def __init__(self, value, label: str = None, content: Widget = None) -> ConditionalItem:
        self.value = value
        self.label = label
        if label is None:
            self.label = str(self.value)
        self.content = content

    def to_json(self):
        return {"label": self.label, "value": self.value}


class DynamicWidget(Widget):

    def __init__(self, widget_id: str = None, file_path: str = __file__):
        self.reload = self.update_template_for_offline_session(self.reload)
        super().__init__(widget_id=widget_id, file_path=file_path)

    def reload(self):
        raise NotImplementedError()

    def update_template_for_offline_session(self, func):
        def wrapper():
            func()
            # to update template for offline session
            from supervisely.app.fastapi.subapp import Application
            os.environ["_SUPERVISELY_OFFLINE_FILES_UPLOADED"] = "False"
            client = Application().test_client
            _ = run_sync(client.get("/"))

        return wrapper


# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523#18425523
# https://ideone.com/ym3bkD
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/13034496/using-global-variables-between-files
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
