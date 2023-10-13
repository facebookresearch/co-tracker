import os
from pathlib import Path
import shlex
import subprocess
from supervisely.io.fs import mkdir
from supervisely.api.api import Api
from supervisely.sly_logger import logger

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

VPN_CONFIGURATION_DIR = "~/supervisely-network"


def supervisely_vpn_network(action: Literal["up", "down"] = "up"):
    # TODO: already down "wg-quick: `wg0' is not a WireGuard interface\n"
    # TODO: wg-quick must be run as root. Please enter the password for max to continue:

    process = subprocess.run(
        shlex.split("curl --max-time 3 -s http://10.8.0.1:80"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )
    if "Connected to Supervisely Net!" in process.stdout:
        logger.info(f"You connected to Supervisely VPN Network")
        return

    api = Api()
    current_dir = Path(__file__).parent.absolute()
    script_path = os.path.join(current_dir, "sly-net.sh")
    network_dir = os.path.expanduser(VPN_CONFIGURATION_DIR)
    mkdir(network_dir)

    process = subprocess.run(
        shlex.split(f"{script_path} {action} {api.token} {api.server_address} {network_dir}"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    text = "connected to"
    if action == "down":
        text = "disconnected from"
    try:
        process.check_returncode()
        logger.info(f"You have been successfully {text} Supervisely VPN Network")
    except subprocess.CalledProcessError as e:
        print(e.stdout)

        e.cmd[2] = "***-api-token-***"
        if "wg0' already exists" in e.stderr:
            logger.info(f"You {text} Supervisely VPN Network")
            pass
        else:
            raise e


def create_debug_task(team_id, port="8000"):
    api = Api()
    me = api.user.get_my_info()
    session_name = me.login + "-development"
    module_id = api.app.get_ecosystem_module_id("supervisely-ecosystem/while-true-script-v2")
    sessions = api.app.get_sessions(team_id, module_id, session_name=session_name)
    redirect_requests = {"token": api.token, "port": port}
    task = None
    for session in sessions:
        if (session.details["meta"].get("redirectRequests") == redirect_requests) and (
            session.details["status"] == str(api.app.Status.QUEUED)
        ):
            task = session.details
            if "id" not in task:
                task["id"] = task["taskId"]
            logger.info(f"Debug task already exists: {task['id']}")
            break
    workspaces = api.workspace.get_list(team_id)
    if task is None:
        task = api.task.start(
            agent_id=None,
            module_id=module_id,
            workspace_id=workspaces[0].id,
            task_name=session_name,
            redirect_requests=redirect_requests,
            proxy_keep_url=False,  # to ignore /net/<token>/endpoint
        )
        if type(task) is list:
            task = task[0]
        logger.info(f"Debug task has been successfully created: {task['taskId']}")
    return task
