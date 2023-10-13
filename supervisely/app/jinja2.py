import typing
from os import PathLike
import jinja2


def create_env(directory: typing.Union[str, PathLike]) -> "jinja2.Environment":
    env_sly = jinja2.Environment(
        loader=jinja2.FileSystemLoader(directory),
        autoescape=True,
        variable_start_string="{{{",
        variable_end_string="}}}",
    )
    return env_sly
