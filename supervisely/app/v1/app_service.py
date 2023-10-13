import json
import os
import time
import traceback
import functools
import sys
import asyncio
import signal
import random
import concurrent.futures
import queue
import re

from supervisely.worker_api.agent_api import AgentAPI
from supervisely.worker_proto import worker_api_pb2 as api_proto
from supervisely.function_wrapper import function_wrapper
from supervisely._utils import take_with_default
from supervisely.sly_logger import logger as default_logger
from supervisely.sly_logger import EventType
from supervisely.app.v1.constants import (
    STATE,
    CONTEXT,
    STOP_COMMAND,
    IMAGE_ANNOTATION_EVENTS,
)
from supervisely.api.api import Api
from supervisely.io.fs import file_exists, mkdir, list_files, get_file_name_with_ext
from supervisely.io.json import load_json_file
from supervisely._utils import _remove_sensitive_information
from supervisely.worker_api.agent_rpc import send_from_memory_generator
from supervisely.io.fs_cache import FileCache


# https://www.roguelynn.com/words/asyncio-we-did-it-wrong/


class ConnectionClosedByServerException(Exception):
    pass


class AppCommandNotFound(Exception):
    pass


REQUEST_ID = "request_id"
SERVER_ADDRESS = "SERVER_ADDRESS"
API_TOKEN = "API_TOKEN"
REQUEST_DATA = "request_data"
AGENT_TOKEN = "AGENT_TOKEN"


def _default_stop(api: Api, task_id, context, state, app_logger):
    app_logger.info("Stop app", extra={"event_type": EventType.APP_FINISHED})


class AppService:
    NETW_CHUNK_SIZE = 1048576
    QUEUE_MAX_SIZE = 2000  # Maximum number of in-flight requests to avoid exhausting server memory.
    DEFAULT_EVENTS = [STOP_COMMAND, *IMAGE_ANNOTATION_EVENTS]

    def __init__(
        self,
        logger=None,
        task_id=None,
        server_address=None,
        agent_token=None,
        ignore_errors=False,
        ignore_task_id=False,
    ):
        self._ignore_task_id = ignore_task_id
        self.logger = take_with_default(logger, default_logger)
        self._ignore_errors = ignore_errors
        self.task_id = take_with_default(task_id, int(os.environ["TASK_ID"]))
        self.server_address = take_with_default(server_address, os.environ[SERVER_ADDRESS])
        self.agent_token = take_with_default(agent_token, os.environ[AGENT_TOKEN])
        self.public_api = Api.from_env(ignore_task_id=self._ignore_task_id)
        self._app_url = self.public_api.app.get_url(self.task_id)
        self._session_dir = "/app"
        self._template_path = None
        debug_app_dir = os.environ.get("DEBUG_APP_DIR", "")
        if debug_app_dir != "":
            self._session_dir = debug_app_dir
        mkdir(self.data_dir)

        self.cache_dir = os.path.join("/apps_cache")
        debug_cache_dir = os.environ.get("DEBUG_CACHE_DIR", "")
        if debug_cache_dir != "":
            self.cache_dir = debug_cache_dir
        mkdir(self.cache_dir)
        self.cache = FileCache(name="FileCache", storage_root=self.cache_dir)

        self.api = AgentAPI(
            token=self.agent_token,
            server_address=self.server_address,
            ext_logger=self.logger,
        )
        self.api.add_to_metadata("x-task-id", str(self.task_id))

        self.callbacks = {}
        self.periodic_items = {}

        self.processing_queue = queue.Queue()  # (maxsize=self.QUEUE_MAX_SIZE)
        self.logger.debug(
            "App is created",
            extra={"task_id": self.task_id, "server_address": self.server_address},
        )

        self._ignore_stop_for_debug = False
        self._error = None
        self.stop_event = asyncio.Event()
        self.has_ui = False

    def _graceful_exit(self, sig, frame):
        asyncio.create_task(self._shutdown(signal=signal.Signals(sig)))

    def _run_executors(self):
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.loop = asyncio.get_event_loop()
        self.logger.trace(f"Operating system: {sys.platform}")
        # May want to catch other signals too
        if os.name == "nt":
            # Windows
            signals = (signal.SIGTERM, signal.SIGINT)
            for s in signals:
                signal.signal(s, self._graceful_exit)
        else:
            # Others
            signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)
            for s in signals:
                self.loop.add_signal_handler(
                    s, lambda s=s: asyncio.create_task(self._shutdown(signal=s))
                )
        # comment out the line below to see how unhandled exceptions behave
        self.loop.set_exception_handler(self.handle_exception)

    def handle_exception(self, loop, context):
        # context["message"] will always be there; but context["exception"] may not
        msg = context.get("exception", context["message"])
        if isinstance(msg, Exception):
            # self.logger.error(traceback.format_exc(), exc_info=True, extra={'exc_str': str(msg), 'future_info': context["future"]})
            self.logger.error(msg, exc_info=True, extra={"future_info": context["future"]})
        else:
            self.logger.error("Caught exception: {}".format(msg))

        self.logger.info("Shutting down...")
        asyncio.create_task(self._shutdown())

    @property
    def session_dir(self):
        return self._session_dir

    @property
    def repo_dir(self):
        return os.path.join(self._session_dir, "repo")

    @property
    def data_dir(self):
        return os.path.join(self._session_dir, "data")

    @property
    def app_url(self):
        from supervisely._utils import abs_url

        return abs_url(self._app_url)

    def _add_callback(self, callback_name, func):
        self.callbacks[callback_name] = func

    def callback(self, callback_name):
        """A decorator that is used to register a view function for a
        given application command.  This does the same thing as :meth:`add_callback`
        but is intended for decorator usage::
            @app.callback('calc')
            def calc_func():
                return 'Hello World'
        :param callback_name: the command name as string
        """

        def decorator(f):
            self._add_callback(callback_name, f)

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                f(*args, **kwargs)

            return wrapper

        return decorator

    def call_periodic_function_sync(self, f, period):
        while True:
            then = time.time()

            try:
                f(api=self.public_api, task_id=self.task_id)
            except Exception as ex:
                tb = traceback.format_exc()
                self.logger.error(f"Exception in periodic function: {f.__name__}\n" f"{tb}")
                self.logger.info("App will be stopped due to error")

                asyncio.run_coroutine_threadsafe(self._shutdown(error=ex), self.loop)

            elapsed = time.time() - then

            if (period - elapsed) > 0:
                # await asyncio.sleep(period - elapsed)
                time.sleep(period - elapsed)

    # async def call_periodic_function(self, period, f):

    async def scheduler(self):
        self.logger.info("Starting scheduler")

        for f, seconds in self.periodic_items.items():
            asyncio.ensure_future(
                self.loop.run_in_executor(
                    self.executor, self.call_periodic_function_sync, f, seconds
                ),
                loop=self.loop,
            )
            # self.loop.create_task(self.call_periodic_function(seconds, f))

    def _add_periodic(self, seconds, f):
        self.periodic_items[f] = seconds

    def periodic(self, seconds):
        """A decorator that is used to call functions periodically
            @app.periodic(seconds=5)
            def log_message_periodically():
                sly.logger.info('periodically message')
        :param seconds: interval of function call in seconds
        """

        def decorator(f):
            self._add_periodic(seconds, f)

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                f(*args, **kwargs)

            return wrapper

        return decorator

    def handle_message_sync(self, request_msg):
        try:
            state = request_msg.get(STATE, None)
            context = request_msg.get(CONTEXT, None)
            if context is not None:
                context["request_id"] = request_msg.get("request_id", "")
            command = request_msg["command"]
            user_api_token = request_msg["api_token"]
            user_public_api = Api(
                self.server_address,
                user_api_token,
                retry_count=5,
                external_logger=self.logger,
                ignore_task_id=self._ignore_task_id,
            )
            self.logger.trace("Event", extra={"request_msg": request_msg})

            if command == STOP_COMMAND:
                self.logger.info("APP receives stop signal from user")
                self.stop_event.set()

            if command == STOP_COMMAND and command not in self.callbacks:
                _default_stop(user_public_api, self.task_id, context, state, self.logger)
                if self._ignore_stop_for_debug is False:
                    # self.stop()
                    asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
                    return
                else:
                    self.logger.info("STOP event is ignored ...")
            elif command in AppService.DEFAULT_EVENTS and command not in self.callbacks:
                raise AppCommandNotFound(
                    'App received default command {!r}. Use decorator "callback" to handle it.'.format(
                        command
                    )
                )
            elif command not in self.callbacks:
                raise AppCommandNotFound(
                    'App received unhandled command {!r}. Use decorator "callback" to handle it.'.format(
                        command
                    )
                )

            if command == STOP_COMMAND:
                if self._ignore_stop_for_debug is False:
                    self.callbacks[command](
                        api=user_public_api,
                        task_id=self.task_id,
                        context=context,
                        state=state,
                        app_logger=self.logger,
                    )
                    asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
                    return
                else:
                    self.logger.info("STOP event is ignored ...")
            else:
                self.callbacks[command](
                    api=user_public_api,
                    task_id=self.task_id,
                    context=context,
                    state=state,
                    app_logger=self.logger,
                )
        except AppCommandNotFound as e:
            self.logger.debug(repr(e), exc_info=False)
        except Exception as e:
            from supervisely.io.exception_handlers import handle_exception

            exception_handler = handle_exception(e)
            if self._ignore_errors is False:
                if exception_handler:
                    # Logging the error and sets the output in Workspace Tasks.
                    exception_handler.log_error_for_agent(command)

                    if self.has_ui:
                        self.show_modal_window(
                            exception_handler.get_message_for_modal_window(),
                            level="error",
                        )

                else:
                    self.logger.error(
                        traceback.format_exc(),
                        exc_info=True,
                        extra={
                            "main_name": command,
                            "exc_str": repr(e),
                            "event_type": EventType.TASK_CRASHED,
                        },
                    )
                self.logger.info("App will be stopped due to error")
                # asyncio.create_task(self._shutdown(error=e))
                asyncio.run_coroutine_threadsafe(self._shutdown(error=e), self.loop)
            else:
                self.logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": repr(e)})
                if self.has_ui:
                    if exception_handler:
                        message = exception_handler.get_message_for_modal_window()
                    else:
                        message = (
                            "Oops! Something went wrong, please try again or contact tech support. "
                            "Find more info in the app logs."
                        )

                    self.show_modal_window(
                        message,
                        level="error",
                    )

    def consume_sync(self):
        while True:
            request_msg = self.processing_queue.get()
            to_log = _remove_sensitive_information(request_msg)
            self.logger.debug("FULL_TASK_MESSAGE", extra={"task_msg": to_log})
            # asyncio.run_coroutine_threadsafe(self.handle_message(request_msg), self.loop)
            asyncio.ensure_future(
                self.loop.run_in_executor(self.executor, self.handle_message_sync, request_msg),
                loop=self.loop,
            )

    async def consume(self):
        self.logger.info("Starting consumer")
        asyncio.ensure_future(
            self.loop.run_in_executor(self.executor, self.consume_sync), loop=self.loop
        )

    def publish_sync(self, initial_events=None):
        if initial_events is not None:
            for event_obj in initial_events:
                event_obj["api_token"] = os.environ[API_TOKEN]
                self.processing_queue.put(event_obj)

        for gen_event in self.api.get_endless_stream(
            "GetGeneralEventsStream", api_proto.GeneralEvent, api_proto.Empty()
        ):
            try:
                data = {}
                if gen_event.data is not None and gen_event.data != b"":
                    data = json.loads(gen_event.data.decode("utf-8"))

                event_obj = {REQUEST_ID: gen_event.request_id, **data}
                self.processing_queue.put(event_obj)
            except Exception as error:
                self.logger.warning("App exception: ", extra={"error_message": repr(error)})

        raise ConnectionClosedByServerException(
            "Requests stream to a deployed model closed by the server."
        )

    async def publish(self, initial_events=None):
        self.logger.info("Starting publisher")
        asyncio.ensure_future(
            self.loop.run_in_executor(self.executor, self.publish_sync, initial_events),
            loop=self.loop,
        )

    def run(self, template_path=None, data=None, state=None, initial_events=None):
        if template_path is None:
            template_path = self.get_template_path()
            self.logger.info(f"App template path: {template_path}")
        else:
            self._template_path = template_path

        if template_path is None:
            template_path = os.path.join(self.repo_dir, "src/gui.html")

        if not file_exists(template_path):
            self.logger.info("App will be running without GUI", extra={"app_url": self.app_url})
            template = ""
        else:
            with open(template_path, "r") as file:
                template = file.read()
            self.has_ui = True

        self.public_api.app.initialize(self.task_id, template, data, state)
        self.logger.info("Application session is initialized", extra={"app_url": self.app_url})

        try:
            self._run_executors()
            self.loop.create_task(self.publish(initial_events), name="Publisher")
            self.loop.create_task(self.consume(), name="Consumer")
            self.loop.create_task(self.scheduler(), name="Scheduler")
            self.loop.run_forever()
        finally:
            self.loop.close()
            self.logger.info("Successfully shutdown the APP service.")

        if self._error is not None:
            raise self._error

    def stop(self, wait=True):
        # @TODO: add timeout
        if wait is True:
            event_obj = {"command": "stop", "api_token": os.environ[API_TOKEN]}
            self.processing_queue.put(event_obj)
        else:
            self.logger.info(
                "Stop app (force, no wait)",
                extra={"event_type": EventType.APP_FINISHED},
            )
            # asyncio.create_task(self._shutdown())
            asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)

    async def _shutdown(self, signal=None, error=None):
        """Cleanup tasks tied to the service's shutdown."""
        if signal:
            self.logger.info(f"Received exit signal {signal.name}...")
        self.logger.info("Nacking outstanding messages")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        [task.cancel() for task in tasks]

        self.logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("Shutting down ThreadPoolExecutor")
        self.executor.shutdown(wait=False)

        self.logger.info(f"Releasing {len(self.executor._threads)} threads from executor")
        for thread in self.executor._threads:
            try:
                thread._tstate_lock.release()
            except Exception:
                pass

        self.logger.info(f"Flushing metrics")
        self.loop.stop()

        if error is not None:
            self._error = error

    def send_response(self, request_id, data):
        out_bytes = json.dumps(data).encode("utf-8")
        self.api.put_stream_with_data(
            "SendGeneralEventData",
            api_proto.Empty,
            send_from_memory_generator(out_bytes, 1048576),
            addit_headers={"x-request-id": request_id},
        )

    def show_modal_window(self, message, level="info"):
        all_levels = ["warning", "info", "error"]
        if level not in all_levels:
            raise ValueError("Unknown level {!r}. Supported levels: {}".format(level, all_levels))

        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message, exc_info=True)

        self.public_api.app.set_field(
            self.task_id, "data.notifyDialog", {"type": level, "message": message}
        )

    def get_template_path(self):
        if self._template_path is not None:
            return self._template_path
        config_path = os.path.join(self.repo_dir, os.environ.get("CONFIG_DIR", ""), "config.json")
        if file_exists(config_path):
            config = load_json_file(config_path)
            self._template_path = config.get("gui_template", None)
            if self._template_path is None:
                self.logger.info("there is no gui_template field in config.json")
            else:
                self._template_path = os.path.join(self.repo_dir, self._template_path)
                if not file_exists(self._template_path):
                    self._template_path = os.path.join(os.path.dirname(sys.argv[0]), "gui.html")
        if self._template_path is None:
            self._template_path = os.path.join(os.path.dirname(sys.argv[0]), "gui.html")
        if file_exists(self._template_path):
            return self._template_path
        return None

    def compile_template(self, root_source_dir=None):
        if root_source_dir is None:
            root_source_dir = self.repo_dir

        def _my_replace_function(match):
            to_replace = match.group(0)
            part_path = match.group(1)
            with open(os.path.join(root_source_dir, part_path), "r") as file:
                part = file.read()
            return part

        template_path = self.get_template_path()
        if template_path is None:
            self.logger.warning("HTML Template for compilation not found")
            return
        with open(template_path, "r") as file:
            template = file.read()

        regex = r"{\%.*include.*'(.*)'.*\%}"
        result = re.sub(regex, lambda m: _my_replace_function(m), template, 0, re.MULTILINE)
        res_path = os.path.join(self.data_dir, "gui.html")
        with open(os.path.join(self.data_dir, "gui.html"), "w") as file:
            file.write(result)
        self._template_path = res_path
        self.logger.info(f"Compiled template is saved to {res_path}")
        return res_path

    def ignore_errors_and_show_dialog_window(self):
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                try:
                    f(*args, **kwargs)
                except Exception as e:
                    from supervisely.io.exception_handlers import handle_exception

                    exception_handler = handle_exception(e)

                    if exception_handler:
                        message = exception_handler.get_message_for_modal_window()
                    else:
                        message = (
                            f"Oops! Something went wrong, please try again or contact tech support."
                            f" Find more info in the app logs. Error: {repr(e)}",
                        )

                    self.logger.error(
                        f"please, contact support: task_id={self.task_id}, {repr(e)}",
                        exc_info=True,
                        extra={
                            "exc_str": str(e),
                        },
                    )
                    self.show_modal_window(
                        message,
                        level="error",
                    )

            return wrapper

        return decorator
