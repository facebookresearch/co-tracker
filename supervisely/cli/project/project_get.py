import traceback

from rich.console import Console

import supervisely as sly


def get_project_name_run(project_id: int) -> bool:
    console = Console()

    api = sly._handle_creds_error_to_console(sly.Api.from_env, console.print)
    if api is False:
        return False

    try:
        project_info = api.project.get_info_by_id(project_id)
        if project_info is None:
            print(
                f"Project with ID={project_id} is either archived, doesn't exist or you don't have enough permissions to access it"
            )
            return False

        print(project_info.name)
        return True

    except:
        traceback.print_exc()
        return False
