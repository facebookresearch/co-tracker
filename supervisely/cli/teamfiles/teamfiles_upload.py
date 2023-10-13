import os
import supervisely as sly
from functools import partial

from tqdm import tqdm
import time
from supervisely.io.fs import get_directory_size

import traceback
from rich.console import Console
from dotenv import load_dotenv
import os


def upload_directory_run(team_id: int, local_dir: str, remote_dir: str) -> bool:
    class MyTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.iteration_value = 0
            self.iteration_number = 0
            self.iteration_locked = False
            self.total_monitor_size = 0

    class MySlyProgress(sly.Progress):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.iteration_value = 0
            self.iteration_number = 0
            self.iteration_locked = False
            self.total_monitor_size = 0

    console = Console()

    api = sly._handle_creds_error_to_console(sly.Api.from_env, console.print)
    if api is False:
        return False

    if api.team.get_info_by_id(team_id) is None:
        console.print(
            f"\nError: Team with ID={team_id} is either doesn't exist or not found in your acocunt\n",
            style="bold red",
        )

    # force directories to end with slash '/'
    if not local_dir.endswith(os.path.sep):
        local_dir = os.path.join(local_dir, "")
    if not remote_dir.endswith("/"):
        remote_dir += "/"

    if not os.path.isdir(local_dir):
        console.print(f"\nError: Local directory '{local_dir}' doesn't exist\n", style="bold red")
        return False

    files = api.file.list2(team_id, remote_dir, recursive=True)
    if len(files) > 0:
        if files[0].path.startswith(remote_dir):
            console.print(
                f"\nError: The Team files folder '{remote_dir}' already exists. Please enter unique path for your folder.\n",
                style="bold red",
            )
            return False
    else:
        pass  # new folder

    console.print(
        f"\nUploading local directory '{local_dir}' to the Team files ...\n",
        style="bold",
    )

    total_size = get_directory_size(local_dir)

    try:
        if sly.is_development():

            def upload_monitor_console(monitor, progress: MyTqdm):
                if progress.n >= progress.total:
                    progress.refresh()
                    progress.close()
                if monitor.bytes_read == 8192:
                    progress.total_monitor_size += monitor.len
                if progress.total_monitor_size > progress.total:
                    progress.total = progress.total_monitor_size
                if not progress.iteration_locked:
                    progress.update(progress.iteration_value + monitor.bytes_read - progress.n)
                if monitor.bytes_read == monitor.len and not progress.iteration_locked:
                    progress.iteration_value += monitor.len
                    progress.iteration_number += 1
                    progress.iteration_locked = True
                    progress.refresh()
                if monitor.bytes_read < monitor.len:
                    progress.iteration_locked = False

            # api.file.upload_directory may be slow depending on the number of folders
            print("Please wait ...")

            progress = MyTqdm(
                desc="Uploading to the Team files...", total=total_size, unit="B", unit_scale=True
            )
            progress_size_cb = partial(upload_monitor_console, progress=progress)

            time.sleep(1)  # for better UX

        else:

            def upload_monitor_instance(monitor, progress: MySlyProgress):
                if monitor.bytes_read == 8192:
                    progress.total_monitor_size += monitor.len
                if progress.total_monitor_size > progress.total:
                    progress.total = progress.total_monitor_size
                if not progress.iteration_locked:
                    progress.set_current_value(
                        progress.iteration_value + monitor.bytes_read, report=False
                    )
                if progress.need_report():
                    progress.report_progress()
                if monitor.bytes_read == monitor.len and not progress.iteration_locked:
                    progress.iteration_value += monitor.len
                    progress.iteration_number += 1
                    progress.iteration_locked = True
                if monitor.bytes_read < monitor.len:
                    progress.iteration_locked = False

            progress = MySlyProgress("Uploading to Team files...", total_size, is_size=True)
            progress_size_cb = partial(upload_monitor_instance, progress=progress)

        # no need in change_name_if_conflict due to previous exception handling
        api.file.upload_directory(
            team_id,
            local_dir,
            remote_dir,
            progress_size_cb=progress_size_cb,
        )

        console.print(
            f"\nLocal directory was sucessfully uploaded to the following path: '{remote_dir}'.\n",
            style="bold green",
        )
        return True

    except:
        console.print("\nUpload failed\n", style="bold red")
        traceback.print_exc()
        return False
