from supervisely.cli.cli import cli
import supervisely as sly
from typing import Callable, Literal, Union


def _handle_creds_error_to_console(
    api_func: sly.Api, printer: Callable = print
) -> Union[sly.Api, Literal[False]]:
    try:
        api = api_func()
    except (KeyError, ValueError) as e:
        if "server_address is not defined" in repr(e):
            text = "SERVER_ADDRESS env variable is undefined. Add it to your '~/supervisely.env' file or to environment variables. Learn more here: https://developer.supervise.ly/getting-started/basics-of-authentication."
            printer(f"\n{text}\n", style="bold red") if "Console.print" in repr(
                printer
            ) else printer(text)
            return False
        elif "api_token is not defined" in repr(e):
            text = "API_TOKEN env variable is undefined. Add it to your '~/supervisely.env' file or to environment variables. Learn more here: https://developer.supervise.ly/getting-started/basics-of-authentication."
            printer(f"\n{text}\n", style="bold red") if "Console.print" in repr(
                printer
            ) else printer(text)
            return False
        else:
            printer(repr(e))
            return False
    return api
