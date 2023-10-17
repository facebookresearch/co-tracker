from functools import wraps
from json import JSONDecodeError, loads
from requests.exceptions import HTTPError, RetryError
from shutil import ReadError
from typing import List, Union, Callable, Dict
from rich.console import Console

from supervisely.sly_logger import logger, EventType
from supervisely.app import DialogWindowError

import traceback
import re

# TODO: Add correct doc link.
# DOC_URL = "https://docs.supervisely.com/errors/"


class HandleException:
    def __init__(
        self,
        exception: Exception,
        stack: List[traceback.FrameSummary] = None,
        **kwargs,
    ):
        self.exception = exception
        self.stack = stack or read_stack_from_exception(self.exception)
        self.code = kwargs.get("code")
        self.title = kwargs.get("title")

        # TODO: Add error code to the title.
        # self.title += f" (SLYE{self.code})"

        self.message = kwargs.get("message")

        # TODO: Add doc link to the message.
        # error_url = DOC_URL + str(self.code)
        # self.message += (
        #     f"<br>Check out the <a href='{error_url}'>documentation</a> "
        #     "for solutions and troubleshooting."
        # )

        self.dev_logging()

    def dev_logging(self):
        # * Option for logging the exception with sly logger.
        # * Due to JSON formatting it's not convinient to read the log.
        # tracelog = " | ".join(traceback.format_list(self.stack))
        # logger.error(
        #     "Detailed info about exception."
        #     f"Exception type: {type(self.exception)}. "
        #     f"Exception value: {self.exception}. "
        #     f"Error code: {self.code}. "
        #     f"Error title: {self.title}. "
        #     f"Error message: {self.message} "
        #     f"Traceback: {tracelog}. "
        # )

        # * Printing the exception to the console line by line.
        console = Console()

        console.print("🛑 Beginning of the error report.", style="bold red")
        console.print(f"🔴 {self.exception.__class__.__name__}: {self.exception}")
        # TODO: Uncomment code line when error codes will be added.
        # console.print(f"Error code: {self.code}.", style="bold orange")
        console.print(f"🔴 Error title: {self.title}")
        console.print(f"🔴 Error message: {self.message}")

        console.print("🔴 Traceback (most recent call last):", style="bold red")

        for i, trace in enumerate(traceback.format_list(self.stack)):
            console.print(f"{i + 1}. {trace}")
        console.print("🛑 End of the error report.", style="bold red")

    def raise_error(self):
        raise DialogWindowError(self.title, self.message)

    def log_error_for_agent(self, main_name: str):
        logger.critical(
            self.title,
            exc_info=True,
            extra={
                "main_name": main_name,
                "event_type": EventType.TASK_CRASHED,
                "exc_str": self.message,
            },
        )

    def get_message_for_modal_window(self):
        return f"{self.title}. \n{self.message}"


class ErrorHandler:
    class APP:
        class UnsupportedShapes(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 1001
                self.title = "Unsupported class shapes"
                self.message = exception.args[0]

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class UnsupportedArchiveFormat(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 1002
                self.title = "Unsupported archive format"
                self.message = "The given archive format is not supported."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class UnicodeDecodeError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 1003
                self.title = "Unicode decode error"
                self.message = (
                    "The given file contains non-unicode characters. "
                    "Please, check the file and try again."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class CallUndeployedModelError(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary] = None, **kwargs
            ):
                self.code = 1004
                self.title = "Call undeployed model error"
                self.message = str(exception.args[0])
                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

    class API:
        class TeamFilesFileNotFound(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2001
                self.title = "Requested file was not found in Team Files"
                self.message = (
                    "The requested file doesn't exist in Team Files in current Team. "
                    "Please, ensure that the file exists and you're working in the correct Team."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class TaskSendRequestError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2002
                self.title = "Task send request error"
                self.message = (
                    "The application has encountered an error while sending a request to the task. "
                    "It may be caused by connection issues. "
                    "Please, check your internet connection, the task is running and input parameters. "
                    "If you are using a model in other session, please, check that the model is serving and it's logs."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class FileSizeTooLarge(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2003
                self.title = "File size limit exceeded"
                self.message = (
                    "The given file size is too large (more than 10 GB) for free community edition."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class ImageFilesSizeTooLarge(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2004
                self.title = "Image files size limit exceeded"
                self.message = "The given image file size is too large (more than 1 GB) for free community edition."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class VideoFilesSizeTooLarge(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2005
                self.title = "Video files size limit exceeded"
                self.message = "The given video file size is too large (more than 300 MB) for free community edition."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class VolumeFilesSizeTooLarge(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2006
                self.title = "Volume files size limit exceeded"
                self.message = "The given volume file size is too large (more than 150 MB) for free community edition."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class OutOfMemory(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2007
                self.title = "Out of memory"
                self.message = (
                    "The agent ran out of memory. "
                    "Please, check your agent's memory usage, reduce batch size or use a device with more memory capacity."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class DockerRuntimeError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2008
                self.title = "Docker runtime error"
                self.message = (
                    "The agent has encountered a Docker runtime error. "
                    "Please, check that docker is installed and running."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class AppSetFieldError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2009
                self.title = "App set field error"
                self.message = (
                    "The application has encountered an error while sending a request. "
                    "It may be caused by connection issues. "
                    "Please, check your internet connection, the app is running and input parameters. "
                    "If you are using a model in other session, please, check that the model is serving and it's logs."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class TeamFilesDirectoryDownloadError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2010
                self.title = "Team files directory download error"
                self.message = "Make sure that the directory exists in the team files, the files are not corrupted, and try again."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class PointcloudsUploadError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2011
                self.title = "Pointclouds uploading error"
                self.message = exception.args[0]

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class AnnotationUploadError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2012
                self.title = "Annotation uploading error"
                self.message = exception.args[0]

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class AnnotationNotFound(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2013
                self.title = "Annotation not found"
                self.message = "Please, check that the annotation(s) exists by the given path."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class ServerOverload(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2014
                self.title = "High load on the server"
                self.message = "Sorry, the server is overloaded. Please, try again later."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class ProjectNotFound(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2015
                self.title = "Project not found"
                self.message = "Please, check that the project exists, not archived and you have enough permissions to access it"

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class DatasetNotFound(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 2016
                self.title = "Dataset not found"
                self.message = "Please, check that the dataset exists, not archived and you have enough permissions to access it."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

    class SDK:
        class ProjectStructureError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 3001
                self.title = "Project structure error"
                self.message = "Please, check the project structure and try again."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class ConversionNotImplemented(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 3002
                self.title = "Not implemented error"
                self.message = (
                    "Conversion is not implemented between the given object types. "
                    "Please, check the geometry type of the objects."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class JsonAnnotationReadError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 3003
                self.title = "JSON annotation read error"
                self.message = (
                    "Please, check that the file has the correct Supervisely JSON format."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class LabelFromJsonFailed(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 3004
                self.title = "Label deserialize error"
                self.message = exception.args[0]

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

    class AgentDocker:
        class ImageNotFound(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                # TODO: rule for error codes inside side apps/modules
                self.code = 4001
                self.title = "Docker image not found"
                default_msg = str(exception)

                try:
                    json_text = exception.args[0].response.text
                    info = loads(json_text)
                    self.message = info.get("message", default_msg)
                except JSONDecodeError:
                    self.message = default_msg

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

    class Agent:
        class AgentError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None):
                self.code = 5001
                self.title = "Agent issue"
                self.message = (
                    "The agent has encountered an error. "
                    "Please, check the agent's logs for more information."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )


ERROR_PATTERNS = {
    AttributeError: {r".*api\.file\.get_info_by_path.*": ErrorHandler.API.TeamFilesFileNotFound},
    HTTPError: {
        r".*api\.task\.send_request.*": ErrorHandler.API.TaskSendRequestError,
        r".*api\.app\.set_field.*": ErrorHandler.API.AppSetFieldError,
        r".*file-storage\.bulk\.upload.*File too large.*": ErrorHandler.API.FileSizeTooLarge,
        r".*images\.bulk\.upload.*FileSize.*\"sizeLimit\":1073741824.*": ErrorHandler.API.ImageFilesSizeTooLarge,
        r".*videos\.bulk\.upload.*FileSize.*sizeLimit\":314572800.*": ErrorHandler.API.VideoFilesSizeTooLarge,
        r".*images\.bulk\.upload.*FileSize.*\"sizeLimit\":157286400.*": ErrorHandler.API.VolumeFilesSizeTooLarge,
        r".*Dataset with datasetId.*is either archived, doesn't exist or you don't have enough permissions to access.*": ErrorHandler.API.DatasetNotFound,
        r".*Project with projectId.*is either archived, doesn't exist or you don't have enough permissions to access.*": ErrorHandler.API.ProjectNotFound,
        r".*api\.task\.set_field.*": ErrorHandler.API.AppSetFieldError,
        r".*Unauthorized for url.*": ErrorHandler.Agent.AgentError,
    },
    RuntimeError: {
        r".*Label\.from_json.*": ErrorHandler.SDK.LabelFromJsonFailed,
        r".*CUDA.*out\sof\smemory.*": ErrorHandler.API.OutOfMemory,
        r"The\ model\ has\ not\ yet\ been\ deployed.*": ErrorHandler.APP.CallUndeployedModelError,
    },
    FileNotFoundError: {
        r".*api\.annotation\.upload_path.*": ErrorHandler.API.AnnotationNotFound,
        r".*api\.annotation\.upload_paths.*": ErrorHandler.API.AnnotationNotFound,
        r".*read_single_project.*": ErrorHandler.SDK.ProjectStructureError,
    },
    JSONDecodeError: {
        r".*sly\.json\.load_json_file.*": ErrorHandler.SDK.JsonAnnotationReadError,
        r".*api\.annotation\.upload_path.*": ErrorHandler.SDK.JsonAnnotationReadError,
        r".*api\.annotation\.upload_paths.*": ErrorHandler.SDK.JsonAnnotationReadError,
    },
    UnicodeDecodeError: {
        r".*codec can't decode byte.*": ErrorHandler.APP.UnicodeDecodeError,
    },
    NotImplementedError: {
        r".*geometry\.convert.*": ErrorHandler.SDK.ConversionNotImplemented,
    },
    ValueError: {
        r".*obj_class\.geometry_type\.from_json.*": ErrorHandler.SDK.LabelFromJsonFailed,
    },
    IndexError: {
        r".*obj_class\.geometry_type\.from_json.*": ErrorHandler.SDK.LabelFromJsonFailed,
    },
    ReadError: {
        r".*shutil\.unpack_archive.*": ErrorHandler.APP.UnsupportedArchiveFormat,
        r".*api\.file\.download_directory.*": ErrorHandler.API.TeamFilesDirectoryDownloadError,
    },
    KeyError: {
        r".*api\.pointcloud\.upload_paths.*": ErrorHandler.API.PointcloudsUploadError,
        r".*api\.pointcloud\.upload_project.*": ErrorHandler.SDK.ProjectStructureError,
        r".*api\.image\.download_bytes.*": ErrorHandler.API.ServerOverload,
        r".*api\.video\.frame\.download_np.*": ErrorHandler.API.ServerOverload,
        r".*api\.image\.download_bytes.*": ErrorHandler.API.ServerOverload,
    },
    TypeError: {
        r".*obj_class\.geometry_type\.from_json.*": ErrorHandler.SDK.LabelFromJsonFailed,
    },
    RetryError: {
        r".*api\.annotation\.upload_paths.*": ErrorHandler.API.AnnotationUploadError,
        r".*api\.task\.set_field.*": ErrorHandler.API.AppSetFieldError,
        r".*api\.app\.set_field.*": ErrorHandler.API.AppSetFieldError,
        r".*api\.task\.send_request.*": ErrorHandler.API.TaskSendRequestError,
    },
}

try:
    from docker.errors import ImageNotFound

    docker_patterns = {
        ImageNotFound: {
            r".*sly\.docker_utils\.docker_pull_if_needed.*": ErrorHandler.AgentDocker.ImageNotFound
        }
    }
except ModuleNotFoundError:
    docker_patterns = {}

ERROR_PATTERNS.update(docker_patterns)


def handle_exception(exception: Exception) -> Union[HandleException, None]:
    """Function for handling exceptions, using the stack trace and patterns for known errors.
    Returns an instance of the ErrorHandler class if the pattern is found, otherwise returns None.

    :param exception: Exception to be handled.
    :type exception: Exception
    :return: Instance of the ErrorHandler class or None.
    :rtype: Union[ErrorHandler, None]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        try:
            # Some code that may raise an exception.
        except Exception as e:
            exception_handler = sly.handle_exception(e)
            if exception_handler:
                # You may raise the exception using the raise_error() method.
                # Or use other approaches for handling the exception (e.g. logging, printing to console, etc.)
                exception_handler.raise_error()
            else:
                # If the pattern is not found, the exception is raised as usual.
                raise
    """
    # Extracting the stack trace.
    stack = read_stack_from_exception(exception)

    # Retrieving the patterns for the given exception type.
    patterns = ERROR_PATTERNS.get(type(exception))
    if not patterns:
        return

    # Looping through the stack trace from the bottom up to find matching pattern with specified Exception type.
    for pattern, handler in patterns.items():
        for frame in stack[::-1]:
            if re.match(pattern, frame.line):
                return handler(exception, stack)
        if re.match(pattern, exception.args[0]):
            return handler(exception, stack)


def handle_exceptions(func: Callable) -> Callable:
    """Decorator for handling exceptions, which tries to find a matching pattern for known errors.
    If the pattern is found, the exception is handled according to the specified handler.
    Otherwise, the exception is raised as usual.

    :param func: Function to be decorated.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        @sly.handle_exceptions
        def my_func():
            # Some code that may raise an exception.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exception_handler = handle_exception(e)
            if exception_handler:
                exception_handler.raise_error()
            else:
                raise

    return wrapper


def handle_additional_exceptions(
    errors: Dict[BaseException, Dict[str, HandleException]] = {},
) -> Callable:
    ERROR_PATTERNS.update(errors)

    def inner(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exception_handler = handle_exception(e)
                if exception_handler:
                    exception_handler.raise_error()
                else:
                    raise

        return wrapper

    return inner


def read_stack_from_exception(exception):
    stack = traceback.extract_stack()
    try:
        stack.extend(traceback.extract_tb(exception.__traceback__))
    except AttributeError:
        pass
    return stack


def get_message_from_exception(exception, title):
    try:
        json_text = exception.args[0].response.text
        info = loads(json_text)
        exc_message = info.get("message", str(exception))
    except:
        exc_message = str(exception)

    return f"{title}: {exc_message}"
