from pathlib import Path
import sys
import os
import json
import traceback
import click
import re
import subprocess
import git
from dotenv import load_dotenv
from rich.console import Console


MIN_SUPPORTED_INSTANCE_VERSION = "6.8.3"

from supervisely.cli.release.release import (
    find_tag_in_repo,
    push_tag,
    get_app_from_instance,
    get_module_root,
    get_module_path,
    get_appKey,
    release,
    slug_is_valid,
    delete_tag,
    get_instance_version,
    get_user,
    get_created_at,
)


def _check_git(repo: git.Repo):
    console = Console()
    result = True

    if len(repo.untracked_files) > 0:
        console.print(
            "[red][Error][/] You have untracked files. Commit all changes before releasing the app."
        )
        console.print("  Untracked files:")
        console.print(
            "\n".join(
                [f"  {i+1}) " + file for i, file in enumerate(repo.untracked_files)][
                    :20
                ]
            )
        )
        if len(repo.untracked_files) > 20:
            console.print(f"  ... and {len(repo.untracked_files) - 20} more.")
        print()
        result = False

    if len(repo.index.diff(None)) > 0:
        console.print(
            "[red][Error][/] You have modified files. Commit all changes before releasing the app."
        )
        console.print("  Modified files:")
        console.print(
            "\n\n".join(
                f"  {i+1}) " + str(d).replace("\n", "\n  ")
                for i, d in enumerate(repo.index.diff(None))
            )
        )
        print()
        result = False

    if len(repo.index.diff("HEAD")) > 0:
        console.print(
            "[red][Error][/] You have staged files. Commit all changes before releasing the app."
        )
        console.print("  Staged files:")
        console.print(
            "\n\n".join(
                f"  {i+1}) " + str(d).replace("\n", "\n  ")
                for i, d in enumerate(repo.index.diff("HEAD"))
            )
        )
        print()
        result = False

    local_branch = repo.active_branch
    remote_branch = local_branch.tracking_branch()
    if remote_branch is None:
        console.print(
            "[red][Error][/] Your branch is not tracking any remote branches. Try running 'git push --set-upstream origin'\n"
        )
        result = False
    elif not local_branch.commit.hexsha == remote_branch.commit.hexsha:
        console.print(
            "[red][Error][/] Local branch and remote branch are different. Push your changes to a remote branch before releasing the app\n"
        )
        result = False

    return result


def _ask_release_version(repo: git.Repo):
    console = Console()

    def extract_version(tag_name):
        return tag_name[len("sly-release-"):] if tag_name.startswith("sly-release-") else tag_name

    try:
        sly_release_tags = [
            tag.name
            for tag in repo.tags
            if _check_release_version(extract_version(tag.name))
        ]
        sly_release_tags.sort(
            key=lambda tag: [int(n) for n in extract_version(tag)[1:].split(".")]
        )
        current_release_version = extract_version(sly_release_tags[-1])[1:]
        suggested_release_version = ".".join(
            [
                *current_release_version.split(".")[:-1],
                str(int(current_release_version.split(".")[-1]) + 1),
            ]
        )
    except Exception:
        current_release_version = None
        suggested_release_version = "0.0.1"
    while True:
        input_msg = f'Enter release version in format vX.X.X ({f"Last release: [blue]v{current_release_version}[/]. " if current_release_version else ""}Press "Enter" for [blue]v{suggested_release_version}[/]):\n'
        release_version = console.input(input_msg)
        if release_version == "":
            release_version = "v" + suggested_release_version
            break
        if _check_release_version(release_version):
            break
        console.print("Wrong version format. Should be of format vX.X.X")

    return release_version


def _ask_release_description():
    console = Console()
    release_description = console.input(
        "Enter release description:\n",
    )
    while release_description.isspace() or release_description == "":
        console.print("Release description cannot be empty")
        release_description = console.input(
            "Enter release description:\n",
        )
    return release_description


def _ask_confirmation():
    while True:
        confirmed = input("Confirm? [y/n]:\n")
        if confirmed.lower() in ["y", "yes"]:
            return True
        if confirmed.lower() in ["n", "no"]:
            return False


def _ask_share_app(server_address):
    console = Console()
    while True:
        confirmed = console.input(
            (
                f"Do you want to share this private app on your private instance [green]{server_address}[/] ?\n"
                f"[green]yes[/] - Application will be available for all users on your private instance.\n"
                "[red]no[/] - Application will only be available for you and the co-authors of the app.\n"
                "You will be able to change the selection on the application page in the ecosystem after publishing. \[y/n]:\n"
            )
        )
        if confirmed.lower() in ["y", "yes"]:
            return True
        if confirmed.lower() in ["n", "no"]:
            return False


def _check_release_version(release_version):
    return re.fullmatch("v\d+\.\d+\.\d+", release_version)


def _check_instance_version(instance_version):
    last_supported = [int(x) for x in MIN_SUPPORTED_INSTANCE_VERSION.split(".")]
    version_numbers = [int(x) for x in instance_version.split(".")]
    for number, supported in zip(version_numbers, last_supported):
        if number > supported:
            return True
        elif number < supported:
            return False
    return True


def get_user_data(server_address, api_token):
    console = Console()
    try:
        user_data = get_user(server_address, api_token)
    except PermissionError:
        console.print(
            "[red][Error][/] Permission denied. Check that all credentials are set correctly"
        )
        return None
    except ConnectionError:
        console.print(
            f'[red][Error][/] Could not access "{server_address}". Check that instance is running and accessible'
        )
        return None
    if user_data is None:
        console.print(
            "[red][Error][/] User not found. Check that all credentials are set correctly"
        )
        return None
    return user_data


def hided(s: str):
    return s[:4] + "*******" + s[-4:]


def run(
    app_directory,
    sub_app_directory,
    slug,
    autoconfirm,
    release_version,
    release_description,
    share,
):
    console = Console()
    console.print("\nSupervisely Release App\n", style="bold")

    # check slug
    if slug and not slug_is_valid(slug):
        console.print("[red][Error][/] Invalid slug")
        return False

    # get module path and check if it is a git repo
    module_root = get_module_root(app_directory)
    if sub_app_directory is not None:
        sub_app_directory = (
            Path(sub_app_directory).absolute().relative_to(module_root).as_posix()
        )
    module_path = get_module_path(module_root, sub_app_directory)
    try:
        repo = git.Repo(module_root)
    except git.InvalidGitRepositoryError:
        console.print(
            f"[red][Error][/] Module path [green]{module_path}[/] is not a git repository"
        )
        return False

    # get server address
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    server_address = os.getenv("SERVER_ADDRESS", None)
    if server_address is None:
        console.print(
            '[red][Error][/] Cannot find [green]SERVER_ADDRESS[/]. Add it to your "~/supervisely.env" file or to environment variables'
        )
        return False

    # get api token
    api_token = os.getenv("API_TOKEN", None)
    if api_token is None:
        console.print(
            '[red][Error][/] Cannot find [green]API_TOKEN[/]. Add it to your "~/supervisely.env" file or to environment variables'
        )
        return False

    # get relese_token and user_id
    user_data = get_user_data(server_address, api_token)
    if user_data is None:
        console.print(
            "[red][Error][/] Cannot find User. Check that all SERVER_ADDERSS and API_TOKEN are set correctly"
        )
        return False
    user_id = user_data["id"]
    user_login = user_data["login"]

    # get relese_token and user_id of releaseing user if needed
    release_token = os.getenv("APP_RELEASE_TOKEN", None)
    release_user_id = None
    release_user_login = None
    if release_token is not None:
        release_user_data = get_user_data(server_address, release_token)
        if release_user_data is None:
            console.print(
                "[red][Error][/] Cannot find Releasing User. Check that all SERVER_ADDERSS and APP_RELEASE_TOKEN are set correctly"
            )
            return False
        release_user_id = release_user_data["id"]
        release_user_login = release_user_data["login"]

    # check instance version
    try:
        instance_version = get_instance_version(api_token, server_address)
        if not _check_instance_version(instance_version["version"]):
            console.print(
                f'[red][Error][/] Instance "{server_address}" does not support releasing apps via sly-release. Please update your instance to version {MIN_SUPPORTED_INSTANCE_VERSION} or higher'
            )
            return False
    except NotImplementedError:
        console.print(
            f'[red][Error][/] Instance "{server_address}" does not support releasing apps via sly-release. Please update your instance to version {MIN_SUPPORTED_INSTANCE_VERSION} or higher'
        )
        return False
    except PermissionError:
        console.print(
            "[red][Error][/] Permission denied. Check that all credentials are set correctly"
        )
        return False
    except ConnectionError:
        console.print(
            f'[red][Error][/] Could not access "{server_address}". Check that instance is running and accessible'
        )
        return False

    # get config
    try:
        with open(module_path.joinpath("config.json"), "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        console.print(
            f'[red][Error][/] Cannot find "config.json" file at "{module_path}"'
        )
        return False
    except json.JSONDecodeError as e:
        console.print(
            f'[red][red][Error][/][/] Cannot decode config json file at "{module_path.joinpath("config.json")}": {str(e)}'
        )
        return False

    # get readme
    try:
        with open(module_path.joinpath("README.md"), "r", encoding="utf_8") as f:
            readme = f.read()
    except:
        readme = ""

    # get app name
    app_name = config.get("name", None)
    if app_name is None:
        console.print(
            f'[red][Error][/] Missing "name" field in config json file at "{module_path.joinpath("config.json")}"'
        )
        return False

    # get modal template
    modal_template = ""
    if "modal_template" in config:
        if config["modal_template"] != "":
            modal_template_path = module_root.joinpath(config["modal_template"])
            if not modal_template_path.exists() or not modal_template_path.is_file():
                console.print(
                    f'[red][Error][/] Cannot find Modal Template at "{modal_template_path}". Please check your [green]config.json[/] file'
                )
                return False
            with open(modal_template_path, "r") as f:
                modal_template = f.read()

    # check that everything is commited and pushed
    success = _check_git(repo)
    if not success:
        return False

    # get appKey
    remote_name = repo.active_branch.tracking_branch().remote_name
    remote = repo.remote(remote_name)
    repo_url = remote.url
    appKey = get_appKey(repo, sub_app_directory, repo_url)

    # check if app exist or not
    app_exist = True
    try:
        app_data = get_app_from_instance(appKey, api_token, server_address)
        if app_data is None:
            app_exist = False
    except PermissionError:
        console.print(
            "[red][Error][/] Permission denied. Check that all credentials are set correctly"
        )
        return False
    except ConnectionError:
        console.print(
            f'[red][Error][/] Could not access "{server_address}". Check that instance is running and accessible'
        )
        return False
    module_exists_label = (
        "[yellow bold]updated[/]" if app_exist else "[green bold]created[/]"
    )

    # print details
    console.print(f"Application directory:\t[green]{module_path}[/]")
    console.print(f"Server address:\t\t[green]{server_address}[/]")
    console.print(f"User Api token:\t\t[green]{hided(api_token)}[/]")
    console.print(f"User:\t\t\t[green]{user_login} (id: {user_id})[/]")
    if release_token:
        console.print(f"Release token:\t\t[green]{hided(release_token)}[/]")
        console.print(
            f"Release User: \t\t[green]{release_user_login} (id: {release_user_id})[/]"
        )
    console.print(f"Git branch:\t\t[green]{repo.active_branch}[/]")
    console.print(f"App Name:\t\t[green]{app_name}[/]")
    console.print(f"App Key:\t\t[green]{hided(appKey)}[/]\n")

    share_app = False
    if not app_exist and server_address.find("app.supervise") == -1:
        if share:
            share_app = True
        elif autoconfirm:
            share_app = True
        else:
            share_app = _ask_share_app(server_address)

    # get and check release version
    if repo.active_branch.name in ["main", "master"]:
        if release_version is None:
            release_version = _ask_release_version(repo)
        if not _check_release_version(release_version):
            console.print(
                '[red][Error][/] Incorrect release version. Should be of format "vX.X.X"'
            )
            return False
    else:
        console.print(
            f'Release version will be "{repo.active_branch.name}" as the branch name'
        )
        release_version = repo.active_branch.name

    # get release name
    if release_description is None:
        release_description = _ask_release_description()

    # print summary
    console.print(
        f'\nApplication "{app_name}" will be {module_exists_label} at "{server_address}" Supervisely instance with release [blue]{release_version}[/] "{release_description}"'
    )
    if not slug:
        if repo.active_branch.name in ["main", "master"]:
            remote_name = repo.active_branch.tracking_branch().name
            console.print(
                f'Git tag "sly-release-{release_version}" will be added and pushed to remote "{remote_name}"'
            )

    # ask for confiramtion if needed
    if autoconfirm:
        confirmed = True
    else:
        confirmed = _ask_confirmation()
    if not confirmed:
        console.print("Release cancelled")
        return False

    # add tag and push
    tag_name = None
    tag_created = False
    if not slug:
        if repo.active_branch.name in ["main", "master"]:
            tag_name = f"sly-release-{release_version}"
            tag = find_tag_in_repo(tag_name, repo)
            if tag is None:
                tag_created = True
                repo.create_tag(tag_name, message=release_description)
            try:
                push_tag(tag_name, repo)
            except subprocess.CalledProcessError:
                if tag_created:
                    repo.delete_tag(tag)
                console.print(
                    f"[red][Error][/] Git push unsuccessful. You need write permissions in repository to release the application"
                )
                return False

    # get created_at
    created_at = get_created_at(repo, tag_name)

    # release
    if release_token:
        api_token = release_token
    console.print("Uploading archive...")
    success = True
    try:
        response = release(
            server_address,
            api_token,
            appKey,
            repo,
            config,
            readme,
            release_description,
            release_version,
            modal_template,
            slug,
            user_id,
            sub_app_directory if sub_app_directory != None else "",
            created_at,
            share_app,
        )
        if response.status_code != 200:
            error = f"[red][Error][/] Error releasing the application. Please contact Supervisely team. Status Code: {response.status_code}"
            try:
                error += ", " + json.dumps(response.json())
            except:
                pass
            console.print(error)
            success = False
        elif response.json()["success"] != True:
            console.print(
                "[red][Error][/] Error releasing the application. Please contact Supervisely team"
                + json.dumps(response.json())
            )
            success = False
    except Exception:
        message = None
        success = False
        console.print(
            f"[red][Error][/] Error releasing the application.\n{traceback.format_exc()}\n"
        )

    # delete tag in case of release failure
    try:
        message = response.json()["details"]["message"]
        if message.startswith("version") and message.endswith("already exists"):
            version_exist = True
        else:
            version_exist = False
    except:
        version_exist = False
    if not success and not version_exist:
        if tag_created:
            console.print(f'Deleting tag "sly-release-{release_version}" from remote')
            try:
                delete_tag(f"sly-release-{release_version}", repo)
            except subprocess.CalledProcessError:
                console.print(
                    f'[orange1][Warning][/] Could not delete tag "sly-release-{release_version}" from remote. Please do it manually to avoid errors'
                )
        return False

    return True


@click.command(
    help="This app allows you to release your aplication to Supervisely platform"
)
@click.option(
    "-p",
    "--path",
    required=False,
    help="[Optional] Path to the directory with application",
)
@click.option(
    "-a",
    "--sub-app",
    required=False,
    help="[Optional] Path to sub-app relative to application directory",
)
@click.option(
    "--release-version",
    required=False,
    help='[Optional] Release version in format "vX.X.X"',
)
@click.option(
    "--release-description", required=False, help="[Optional] Release description"
)
@click.option(
    "--share",
    is_flag=True,
    help="[Optional] Add this flag to share the private app on your private instance",
)
@click.option("-y", is_flag=True, help="[Optional] Add this flag for autoconfirm")
@click.option("-s", "--slug", required=False, help="[Optional] For internal use")
def cli_run(path, sub_app, slug, y, release_version, release_description, share):
    try:
        success = run(
            app_directory=path,
            sub_app_directory=sub_app,
            slug=slug,
            autoconfirm=y,
            release_version=release_version,
            release_description=release_description,
            share=share,
        )
        if success:
            print("App released sucessfully!")
            sys.exit(0)
        else:
            print("App not released")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Aborting...")
        print("App not released")
        sys.exit(1)
