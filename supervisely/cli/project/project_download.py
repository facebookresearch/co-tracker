import traceback

from rich.console import Console
import tqdm
import supervisely as sly


def download_run(id: int, dest_dir: str) -> bool:
    console = Console()

    api = sly._handle_creds_error_to_console(sly.Api.from_env, console.print)
    if api is False:
        return False

    console.print(
        f"\nDownloading data from project with ID={id} to directory: '{dest_dir}' ...\n",
        style="bold",
    )

    project_info = api.project.get_info_by_id(id)
    if project_info is None:
        console.print(f"\nError: Project '{project_info}' doesn't exists\n", style="bold red")
        return False

    n_count = project_info.items_count
    try:
        with tqdm.tqdm(total=n_count) as pbar:
            sly.download(api, id, dest_dir, progress_cb=pbar)

        console.print(
            f"\nProject '{project_info.name}' has been downloaded sucessfully!\n",
            style="bold green",
        )
        return True
    except:
        console.print(
            f"\nProject '{project_info.name}' has been failed to download\n", style="bold red"
        )
        traceback.print_exc()
        return False
