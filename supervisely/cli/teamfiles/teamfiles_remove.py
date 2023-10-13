import traceback

import supervisely as sly
from rich.console import Console
from dotenv import load_dotenv
import os


def remove_file_run(team_id: int, path: str) -> bool:
    console = Console()

    api = sly._handle_creds_error_to_console(sly.Api.from_env, console.print)
    if api is False:
        return False

    if api.team.get_info_by_id(team_id) is None:
        console.print(f"\nError: Team with ID={team_id} not exists\n", style="bold red")
        return False

    try:
        api.file.remove_file(team_id, path)
        console.print(f"\nFile '{path}' was successfully removed\n", style="bold green")
        return True

    except:
        console.print(f"\nRemoving file failed\n", style="bold red")
        traceback.print_exc()
        return False


def remove_directory_run(team_id: int, path: str) -> bool:
    console = Console()

    api = sly._handle_creds_error_to_console(sly.Api.from_env, console.print)
    if api is False:
        return False

    if api.team.get_info_by_id(team_id) is None:
        console.print(f"\nError: Team with ID={team_id} doesn't exist\n", style="bold red")
        return False

    try:
        api.file.remove_dir(team_id, path)
        console.print(f"\nDirectory '{path}' has been successfully removed\n", style="bold green")
        return True

    except:
        console.print(f"\nError: Removing directory has been failed\n", style="bold red")
        traceback.print_exc()
        return False
