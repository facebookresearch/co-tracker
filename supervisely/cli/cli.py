import sys
import click


from supervisely.cli.project import download_run, upload_run, get_project_name_run
from supervisely.cli.task import set_output_directory_run
from supervisely.cli.teamfiles import (
    remove_file_run,
    remove_directory_run,
    upload_directory_run,
    download_directory_run,
)


@click.group()
def cli():
    pass


@cli.command(help="This app allows you to release your aplication to Supervisely platform")
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
    "--release-description",
    required=False,
    help="[Optional] Release description (max length is 64 symbols)",
)
@click.option(
    "--share",
    is_flag=True,
    help="[Optional] Add this flag to share the private app on your private instance",
)
@click.option("-y", is_flag=True, help="[Optional] Add this flag for autoconfirm")
@click.option("-s", "--slug", required=False, help="[Optional] For internal use")
def release(path, sub_app, slug, y, release_version, release_description, share):
    from supervisely.cli.release import run

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


@cli.group()
def project():
    """Commands: download, get-name, upload"""
    pass


@project.command(help="Download project data from supervisely to local directory")
@click.option(
    "-id",
    "--id",
    required=True,
    type=int,
    help="Supervisely project ID",
)
@click.option(
    "-d",
    "--dst",
    required=True,
    type=str,
    help="Download destination directory",
)
def download(id: int, dst: str) -> None:
    try:
        success = download_run(id, dst)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload aborted\n")
        sys.exit(1)


@project.command(help="Get project name")
@click.option(
    "-id",
    "--id",
    required=True,
    type=int,
    help="Supervisely project ID",
)
def get_name(id: int) -> None:
    try:
        success = get_project_name_run(id)
        if success:
            sys.exit(0)
        else:
            print("Getting project name failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Getting project name directory aborted")
        sys.exit(1)


@project.command(help="Upload project data from local directory")
@click.option(
    "-s",
    "--src",
    required=True,
    type=str,
    help="Upload source directory",
)
@click.option(
    "-id",
    "--id",
    required=True,
    type=int,
    help="Destination supervisely workspace ID",
)
@click.option(
    "-n",
    "--name",
    required=False,
    type=str,
    help="Custom project name",
)
def upload(src: str, id: int, name: str) -> None:
    try:
        success = upload_run(src, id, name)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload aborted\n")
        sys.exit(1)


@cli.group()
def teamfiles():
    """Commands: download, remove-file, remove-dir, upload"""
    pass


@teamfiles.command(
    help="Download source files from Team files directory with destination to local path"
)
@click.option(
    "-id",
    "--id",
    required=True,
    type=int,
    help="Supervisely team ID",
)
@click.option(
    "-s",
    "--src",
    required=True,
    type=str,
    help="Path to Team files source directory from which files are downloaded",
)
@click.option(
    "-d",
    "--dst",
    required=True,
    type=str,
    help="Path to local destination directory to which files are downloaded",
)
@click.option(
    "-f",
    "--filter",
    required=False,
    type=str,
    help="[Optional] Filter downloaded files using regexp",
)
@click.option(
    "-i",
    is_flag=True,
    help="[Optional] Ignore and skip if source directory not exists",
)
def download(id: int, src: str, dst: str, filter: str, i: bool) -> None:
    try:
        success = download_directory_run(id, src, dst, filter, ignore_if_not_exists=i)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("Download aborted")
        sys.exit(1)


@teamfiles.command(help="Remove file from supervisely teamfiles")
@click.option(
    "-id",
    "--id",
    required=True,
    type=int,
    help="Supervisely team ID",
)
@click.option(
    "-p",
    "--path",
    required=True,
    type=str,
    help="File path to remove",
)
def remove_file(id: int, path: str) -> None:
    try:
        success = remove_file_run(id, path)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nRemoving file aborted\n")
        sys.exit(1)


@teamfiles.command(help="Remove directory from supervisely Team files")
@click.option(
    "-id",
    "--id",
    required=True,
    type=int,
    help="Supervisely team ID",
)
@click.option(
    "-p",
    "--path",
    required=True,
    type=str,
    help="Path to remove directory",
)
def remove_dir(id: int, path: str) -> None:
    try:
        success = remove_directory_run(id, path)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("Removing directory aborted")
        sys.exit(1)


@teamfiles.command(help="Upload local source files with destination to supervisely Team files")
@click.option(
    "-id",
    "--id",
    required=True,
    type=int,
    help="Supervisely team ID",
)
@click.option(
    "-s",
    "--src",
    required=True,
    type=str,
    help="Path to local source directory from which files are uploaded",
)
@click.option(
    "-d",
    "--dst",
    required=True,
    type=str,
    help="Path to Team files remote destination directory to which files are uploaded",
)
def upload(id: int, src: str, dst: str) -> None:
    try:
        success = upload_directory_run(id, src, dst)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("Upload aborted")
        sys.exit(1)


@cli.group()
def task():
    """Commands: set-output-dir"""
    pass


@task.command(help="Set link to Team files directory at workspace tasks interface")
@click.option(
    "-id",
    "--id",
    required=False,
    type=int,
    help="[Optional] Supervisely task ID",
)
@click.option(
    "--team-id",
    required=False,
    type=int,
    help="[Optional] Supervisely team ID",
)
@click.option(
    "-d",
    "--dir",
    required=True,
    type=str,
    help="Path to Team files directory",
)
def set_output_dir(id: int, team_id: int, dir: str) -> None:
    try:
        success = set_output_directory_run(id, team_id, dir)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("Setting task output directory aborted")
        sys.exit(1)
