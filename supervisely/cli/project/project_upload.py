import traceback

from rich.console import Console
import tqdm

import supervisely as sly
from supervisely.io.fs import dir_exists


def upload_run(src_dir: str, workspace_id: int, project_name: str = None) -> bool:
    console = Console()

    api = sly._handle_creds_error_to_console(sly.Api.from_env, console.print)
    if api is False:
        return False

    project_fs = sly.read_project(src_dir)
    if project_name is None:
        project_name = project_fs.name

    console.print(
        f"\nUploading data from the source directory: '{src_dir}' ...\n",
        style="bold",
    )

    if not dir_exists(src_dir):
        console.print(f"\nError: Directory '{src_dir}' doesn't exist\n", style="bold red")
        return False

    if api.workspace.get_info_by_id(workspace_id) is None:
        console.print(
            f"\nError: Workspace with id={workspace_id} doesn't exist\n", style="bold red"
        )

    try:
        with tqdm.tqdm(total=project_fs.total_items) as pbar:
            sly.upload(src_dir, api, workspace_id, project_name, progress_cb=pbar)
            pbar.refresh()
        console.print(
            f"\nProject '{project_name}' has been uploaded sucessfully!\n", style="bold green"
        )
        return True
    except:
        console.print(f"\nProject '{project_name}' has been failed to upload\n", style="bold red")
        traceback.print_exc()
        return False
