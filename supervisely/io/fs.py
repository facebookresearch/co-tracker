# coding: utf-8

# docs
from re import L
from typing import Dict, List, Optional, Callable, Union, Literal, Generator

import os
import re
import shutil
import errno
import tarfile
import subprocess
import requests
from requests.structures import CaseInsensitiveDict
from collections.abc import Mapping

from tqdm import tqdm

from supervisely._utils import get_bytes_hash, get_string_hash
from supervisely.io.fs_cache import FileCache
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress

JUNK_FILES = [".DS_Store", "__MACOSX", "._.DS_Store", "Thumbs.db", "desktop.ini"]


def get_file_name(path: str) -> str:
    """
    Extracts file name from a given path.

    :param path: Path to file.
    :type path: str
    :returns: File name without extension
    :rtype: :class:`str`
    :Usage example:

     .. code-block::

        import supervisely as sly

        file_name = sly.fs.get_file_name("/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg")

        print(file_name)
        # Output: IMG_0748
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_file_ext(path: str) -> str:
    """
    Extracts file extension from a given path.

    :param path: Path to file.
    :type path: str
    :returns: File extension without name
    :rtype: :class:`str`
    :Usage example:

     .. code-block::

        import supervisely as sly

        file_ext = sly.fs.get_file_ext("/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg")

        print(file_ext)
        # Output: .jpeg
    """
    return os.path.splitext(os.path.basename(path))[1]


def get_file_name_with_ext(path: str) -> str:
    """
    Extracts file name with ext from a given path.

    :param path: Path to file.
    :type path: str
    :returns: File name with extension
    :rtype: :class:`str`
    :Usage example:

     .. code-block::

        import supervisely as sly

        file_name_ext = sly.fs.get_file_name_with_ext("/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg")

        print(file_name_ext)
        # Output: IMG_0748.jpeg
    """
    return os.path.basename(path)


def remove_junk_from_dir(dir: str) -> List[str]:
    """
    Cleans the given directory from junk files and dirs (e.g. .DS_Store, __MACOSX, Thumbs.db, etc.).

    :param dir: Path to directory.
    :type dir: str
    :returns: List of global paths to removed files and dirs.
    :rtype: List[str]

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        input_dir = "/home/admin/work/projects/lemons_annotated/"
        sly.fs.remove_junk_from_dir(input_dir)
    """
    paths = list_dir_recursively(dir, include_subdirs=True, use_global_paths=True)
    removed_paths = []
    for path in paths:
        if get_file_name(path) in JUNK_FILES:
            if os.path.isfile(path):
                silent_remove(path)
                removed_paths.append(path)
            elif os.path.isdir(path):
                remove_dir(path)
                removed_paths.append(path)


def list_dir_recursively(
    dir: str, include_subdirs: bool = False, use_global_paths: bool = False
) -> List[str]:
    """
    Recursively walks through directory and returns list with all file paths.

    :param path: Path to directory.
    :type path: str
    :param include_subdirs: If True, subdirectories will be included in the result list.
    :type include_subdirs: bool
    :param use_global_paths: If True, absolute paths will be returned instead of relative ones.
    :type use_global_paths: bool
    :returns: List containing file paths.
    :rtype: :class:`List[str]`
    :Usage example:

     .. code-block::

        import supervisely as sly

        list_dir = sly.fs.list_dir_recursively("/home/admin/work/projects/lemons_annotated/")

        print(list_dir)
        # Output: ['meta.json', 'ds1/ann/IMG_0748.jpeg.json', 'ds1/ann/IMG_4451.jpeg.json', 'ds1/img/IMG_0748.jpeg', 'ds1/img/IMG_4451.jpeg']
    """
    all_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            file_path = os.path.join(root, name)
            file_path = (
                os.path.relpath(file_path, dir)
                if not use_global_paths
                else os.path.abspath(file_path)
            )
            all_files.append(file_path)
        if include_subdirs:
            for name in dirs:
                dir_path = os.path.join(root, name)
                dir_path = (
                    os.path.relpath(dir_path, dir)
                    if not use_global_paths
                    else os.path.abspath(dir_path)
                )
                all_files.append(dir_path)
    return all_files


def list_files_recursively(
    dir: str, valid_extensions: Optional[List[str]] = None, filter_fn=None
) -> List[str]:
    """
    Recursively walks through directory and returns list with all file paths.

     :param dir: Target dir path.
     :param dir: str
     :param valid_extensions: List with valid file extensions.
     :type valid_extensions: List[str]
     :param filter_fn: Function with a single argument that determines whether to keep a given file path.
     :type filter_fn:
     :returns: List with file paths
     :rtype: :class:`List[str]`
     :Usage example:

      .. code-block:: python

         import supervisely as sly

         list_files = sly.fs.list_files_recursively("/home/admin/work/projects/lemons_annotated/ds1/img/")

         print(list_files)
         # Output: ['/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg', '/home/admin/work/projects/lemons_annotated/ds1/img/IMG_4451.jpeg']
    """

    def file_path_generator():
        for dir_name, _, file_names in os.walk(dir):
            for filename in file_names:
                yield os.path.join(dir_name, filename)

    return [
        file_path
        for file_path in file_path_generator()
        if (valid_extensions is None or get_file_ext(file_path) in valid_extensions)
        and (filter_fn is None or filter_fn(file_path))
    ]


def list_files(dir: str, valid_extensions: Optional[List[str]] = None, filter_fn=None) -> List[str]:
    """
    Returns list with file paths presented in given directory.

    :param dir: Target dir path.
    :param dir: str
    :param valid_extensions: List with valid file extensions.
    :type valid_extensions: List[str]
    :param filter_fn: Function with a single argument that determines whether to keep a given file path.
    :type filter_fn:
    :returns: List with file paths
    :rtype: :class:`List[str]`
    :Usage example:

     .. code-block:: python

         import supervisely as sly

         list_files = sly.fs.list_files("/home/admin/work/projects/lemons_annotated/ds1/img/")

         print(list_files)
         # Output: ['/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg', '/home/admin/work/projects/lemons_annotated/ds1/img/IMG_4451.jpeg']
    """
    res = list(os.path.join(dir, x.name) for x in os.scandir(dir) if x.is_file())
    return [
        file_path
        for file_path in res
        if (valid_extensions is None or get_file_ext(file_path) in valid_extensions)
        and (filter_fn is None or filter_fn(file_path))
    ]


def mkdir(dir: str, remove_content_if_exists: Optional[bool] = False) -> None:
    """
    Creates a leaf directory and all intermediate ones.

    :param dir: Target dir path.
    :param dir: str
    :remove_content_if_exists: Remove directory content if it exist.
    :remove_content_if_exists: bool
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import mkdir
        mkdir('/home/admin/work/projects/example')
    """
    if dir_exists(dir) and remove_content_if_exists is True:
        clean_dir(dir, ignore_errors=True)
    else:
        os.makedirs(dir, mode=0o777, exist_ok=True)


# create base path recursively; pay attention to slash-terminating paths
def ensure_base_path(path: str) -> None:
    """
    Recursively create parent directory for target path.

    :param path: Target dir path.
    :type path: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import ensure_base_path
        ensure_base_path('/home/admin/work/projects/example')
    """
    dst_dir = os.path.split(path)[0]
    if dst_dir:
        mkdir(dst_dir)


def copy_file(src: str, dst: str) -> None:
    """
    Copy file from one path to another, if destination directory doesn't exist it will be created.

    :param src: Source file path.
    :type src: str
    :param dst: Destination file path.
    :type dst: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import copy_file
        copy_file('/home/admin/work/projects/example/1.png', '/home/admin/work/tests/2.png')
    """
    ensure_base_path(dst)
    with open(dst, "wb") as out_f:
        with open(src, "rb") as in_f:
            shutil.copyfileobj(in_f, out_f, length=1024 * 1024)


def hardlink_or_copy_file(src: str, dst: str) -> None:
    """
    Creates a hard link pointing to src named dst. If the link cannot be created, the file will be copied.

    :param src: Source file path.
    :type src: str
    :param dst: Destination file path.
    :type dst: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import hardlink_or_copy_file
        hardlink_or_copy_file('/home/admin/work/projects/example/1.png', '/home/admin/work/tests/link.txt')
    """
    try:
        os.link(src, dst)
    except OSError:
        copy_file(src, dst)


def hardlink_or_copy_tree(src: str, dst: str) -> None:
    """
    Creates a hard links pointing to src named dst files recursively. If the link cannot be created, the file will be copied.

    :param src: Source dir path.
    :type src: str
    :param dst: Destination dir path.
    :type dst: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import hardlink_or_copy_tree
        hardlink_or_copy_tree('/home/admin/work/projects/examples', '/home/admin/work/tests/links')
    """
    mkdir(dst)
    for dir_name, _, file_names in os.walk(src):
        relative_dir = os.path.relpath(dir_name, src)
        dst_sub_dir = os.path.join(dst, relative_dir)
        mkdir(dst_sub_dir)
        for file_name in file_names:
            hardlink_or_copy_file(
                os.path.join(dir_name, file_name), os.path.join(dst_sub_dir, file_name)
            )


def dir_exists(dir: str) -> bool:
    """
    Check whether directory exists or not.

    :param dir: Target directory path.
    :type dir: str
    :returns: True if directory exists, False otherwise.
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

          from supervisely.io.fs import dir_exists
          dir_exists('/home/admin/work/projects/examples') # True
          dir_exists('/home/admin/work/not_exist_dir') # False
    """
    return os.path.isdir(dir)


def dir_empty(dir: str) -> bool:
    """
    Check whether directory is empty or not.

    :param dir: Target directory path.
    :type dir: str
    :returns: True if directory is empty, False otherwise.
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

          from supervisely.io.fs import dir_empty
          dir_empty('/home/admin/work/projects/examples') # False
    """
    if dir_exists(dir) and len(list_files_recursively(dir)) > 0:
        return False
    return True


def file_exists(path: str) -> bool:
    """
    Check whether file exists or not.

    :param dir: Target file path.
    :type dir: str
    :returns: True if file exists, False otherwise.
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

          from supervisely.io.fs import file_exists
          file_exists('/home/admin/work/projects/examples/1.jpeg') # True
          file_exists('/home/admin/work/projects/examples/not_exist_file.jpeg') # False
    """
    return os.path.isfile(path)


def get_subdirs(dir_path: str) -> list:
    """
    Get list containing the names of the directories in the given directory.

    :param dir_path: Target directory path.
    :type dir_path: str
    :returns: List containing directories names.
    :rtype: :class:`list`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import get_subdirs
        subdirs = get_subdirs('/home/admin/work/projects/examples')
        print(subdirs)
        # Output: ['tests', 'users', 'ds1']
    """
    res = list(x.name for x in os.scandir(dir_path) if x.is_dir())
    return res


# removes directory content recursively
def clean_dir(dir_: str, ignore_errors: Optional[bool] = True) -> None:
    """
    Recursively delete a directory tree, but save root directory.

    :param dir_: Target directory path.
    :type dir_: str
    :ignore_errors: Ignore possible errors while removes directory content.
    :ignore_errors: bool
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import clean_dir
        clean_dir('/home/admin/work/projects/examples')
    """
    # old implementation
    # shutil.rmtree(dir_, ignore_errors=True)
    # mkdir(dir_)

    for filename in os.listdir(dir_):
        file_path = os.path.join(dir_, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.warn(f"Failed to delete {file_path}. Reason: {repr(e)}")
            if ignore_errors is False:
                raise e


def remove_dir(dir_: str) -> None:
    """
    Recursively delete a directory tree.

    :param dir_: Target directory path.
    :type dir_: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import remove_dir
        remove_dir('/home/admin/work/projects/examples')
    """
    shutil.rmtree(dir_, ignore_errors=True)


def silent_remove(file_path: str) -> None:
    """
    Remove file which may not exist.

    :param file_path: File path.
    :type file_path: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import silent_remove
        silent_remove('/home/admin/work/projects/examples/1.jpeg')
    """
    try:
        os.remove(file_path)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise


def get_file_size(path: str) -> int:
    """
    Get the size of a file.

    :param path: File path.
    :type path: str
    :returns: File size in bytes
    :rtype: :class:`int`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import get_file_size
        file_size = get_file_size('/home/admin/work/projects/examples/1.jpeg') # 161665
    """
    return os.path.getsize(path)


def get_directory_size(dir_path: str) -> int:
    """
    Get the size of a directory.

    :param path: Target directory path.
    :type path: str
    :returns: Directory size in bytes
    :rtype: :class:`int`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import get_directory_size
        dir_size = get_directory_size('/home/admin/work/projects/examples') # 8574563
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += get_file_size(fp)
    return total_size


def archive_directory(
    dir_: str, tar_path: str, split: Optional[Union[int, str]] = None
) -> Union[None, List[str]]:
    """
    Create tar archive from directory and optionally split it into parts of specified size.

    :param dir_: Target directory path.
    :type dir_: str
    :param tar_path: Path for output tar archive.
    :type tar_path: str
    :param split: Split archive into parts of specified size (in bytes) or size with
        suffix (e.g. '1Kb' = 1024, '1Mb' = 1024 * 1024). Default is None.
    :type split: Union[int, str]
    :returns: None or list of archive parts if split is not None
    :rtype: Union[None, List[str]]
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import archive_directory
        # If split is not needed.
        archive_directory('/home/admin/work/projects/examples', '/home/admin/work/examples.tar')

        # If split is specified.
        archive_parts_paths = archive_directory('/home/admin/work/projects/examples', '/home/admin/work/examples/archive.tar', split=1000000)
        print(archive_parts_paths) # ['/home/admin/work/examples/archive.tar.001', '/home/admin/work/examples/archive.tar.002']
    """
    with tarfile.open(tar_path, "w", encoding="utf-8") as tar:
        tar.add(dir_, arcname=os.path.sep)

    if split is None:
        return

    split = string_to_byte_size(split)

    if os.path.getsize(tar_path) <= split:
        return

    tar_name = os.path.basename(tar_path)
    tar_dir = os.path.abspath(os.path.dirname(tar_path))
    parts_paths = []
    part_number = 1

    with open(tar_path, "rb") as input_file:
        while True:
            part_name = f"{tar_name}.{str(part_number).zfill(3)}"
            output_path = os.path.join(tar_dir, part_name)
            data = input_file.read(split)
            if not data:
                break
            with open(output_path, "wb") as output_file:
                output_file.write(data)
                parts_paths.append(output_path)
                part_number += 1

    os.remove(tar_path)
    return parts_paths


def unpack_archive(archive_path: str, target_dir: str, remove_junk=True) -> None:
    """
    Unpacks archive to the target directory, removes junk files and directories.

    :param archive_path: Path to the archive.
    :type archive_path: str
    :param target_dir: Path to the target directory.
    :type target_dir: str
    :param remove_junk: Remove junk files and directories. Default is True.
    :type remove_junk: bool
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        archive_path = '/home/admin/work/examples.tar'
        target_dir = '/home/admin/work/projects'

        sly.fs.unpack_archive(archive_path, target_dir)
    """
    shutil.unpack_archive(archive_path, target_dir)
    if remove_junk:
        remove_junk_from_dir(target_dir)


def string_to_byte_size(string: Union[str, int]) -> int:
    """Returns integer representation of byte size from string representation.
        If input is integer, returns the same integer for convenience.

        :param string: string representation of byte size (e.g. 1.5Kb, 2Mb, 3.7Gb, 4.2Tb) or integer
        :type string: Union[str, int]
        :return: integer representation of byte size (or the same integer if input is integer)
        :rtype: int

        :raises ValueError: if input string is invalid

    :Usage example:

    .. code-block:: python
        string_size = "1.5M"
        size = string_to_byte_size(string_size)
        print(size)  # 1572864

    """

    MULTIPLIER = 1024
    units = {"KB": 1, "MB": 2, "GB": 3, "TB": 4}

    if isinstance(string, int):
        return string

    try:
        value, unit = string[:-2], string[-2:].upper()
        multiplier = MULTIPLIER ** units[unit]
        return int(float(value) * multiplier)
    except (KeyError, ValueError, IndexError):
        raise ValueError(
            "Invalid input string. The string must be in the format of '1.5Kb', '2Mb', '3.7Gb', '4.2Tb' or integer."
        )


def get_file_hash(path: str) -> str:
    """
    Get hash from target file.

    :param path: Target file path.
    :type path: str
    :returns: File hash
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import get_file_hash
        hash = get_file_hash('/home/admin/work/projects/examples/1.jpeg') # rKLYA/p/P64dzidaQ/G7itxIz3ZCVnyUhEE9fSMGxU4=
    """
    return get_bytes_hash(open(path, "rb").read())


def tree(dir_path: str) -> str:
    """
    Get tree for target directory.

    :param dir_path: Target directory path.
    :type dir_path: str
    :returns: Tree with directory files and subdirectories
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import tree
        dir_tree = tree('/home/admin/work/projects/examples')
        print(dir_tree)
        # Output: /home/admin/work/projects/examples
        # ├── [4.0K]  1
        # │   ├── [165K]  crop.jpeg
        # │   ├── [169K]  fliplr.jpeg
        # │   ├── [169K]  flipud.jpeg
        # │   ├── [166K]  relative_crop.jpeg
        # │   ├── [167K]  resize.jpeg
        # │   ├── [169K]  rotate.jpeg
        # │   ├── [171K]  scale.jpeg
        # │   └── [168K]  translate.jpeg
        # ├── [ 15K]  123.jpeg
        # ├── [158K]  1.jpeg
        # ├── [188K]  1.txt
        # ├── [1.3M]  1.zip
        # ├── [4.0K]  2
        # ├── [ 92K]  acura.png
        # ├── [1.2M]  acura_PNG122.png
        # ├── [198K]  aston_martin_PNG55.png
        # ├── [4.0K]  ds1
        # │   ├── [4.0K]  ann
        # │   │   ├── [4.3K]  IMG_0748.jpeg.json
        # │   │   ├── [ 151]  IMG_0777.jpeg.json
        # │   │   ├── [ 151]  IMG_0888.jpeg.json
        # │   │   ├── [3.7K]  IMG_1836.jpeg.json
        # │   │   ├── [8.1K]  IMG_2084.jpeg.json
        # │   │   ├── [5.5K]  IMG_3861.jpeg.json
        # │   │   ├── [6.0K]  IMG_4451.jpeg.json
        # │   │   └── [5.0K]  IMG_8144.jpeg.json
        # │   └── [4.0K]  img
        # │       ├── [152K]  IMG_0748.jpeg
        # │       ├── [210K]  IMG_0777.jpeg
        # │       ├── [210K]  IMG_0888.jpeg
        # │       ├── [137K]  IMG_1836.jpeg
        # │       ├── [139K]  IMG_2084.jpeg
        # │       ├── [145K]  IMG_3861.jpeg
        # │       ├── [133K]  IMG_4451.jpeg
        # │       └── [136K]  IMG_8144.jpeg
        # ├── [152K]  example.jpeg
        # ├── [2.4K]  example.json
        # ├── [153K]  flip.jpeg
        # ├── [ 65K]  hash1.jpeg
        # ├── [ 336]  meta.json
        # └── [5.4K]  q.jpeg
        # 5 directories, 37 files
    """
    out = subprocess.Popen(
        ["tree", "--filelimit", "500", "-h", "-n", dir_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, stderr = out.communicate()
    return stdout.decode("utf-8")


def log_tree(
    dir_path: str,
    logger,
    level: Literal["info", "debug", "warning", "error"] = "info",
) -> None:
    """
    Get tree for target directory and displays it in the log.

    :param dir_path: Target directory path.
    :type dir_path: str
    :param logger: Logger to display data.
    :type logger: logger
    :type level: Logger level. Available levels: info, debug, warning, error. Default: info.
    :type level: Literal["info", "debug", "warning", "error"]
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import log_tree
        logger = sly.logger
        log_tree('/home/admin/work/projects/examples', logger)
    """
    out = tree(dir_path)

    log_levels = {
        "info": logger.info,
        "debug": logger.debug,
        "warning": logger.warning,
        "error": logger.error,
    }
    if level not in log_levels:
        raise ValueError(
            f"Unknown logger level: {level}. Available levels: info, debug, warning, error"
        )
    log_func = log_levels[level]
    log_func("DIRECTORY_TREE", extra={"tree": out})


def touch(path: str) -> None:
    """
    Sets access and modification times for a file.

    :param path: Target file path.
    :type path: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import touch
        touch('/home/admin/work/projects/examples/1.jpeg')
    """
    ensure_base_path(path)
    with open(path, "a"):
        os.utime(path, None)


def download(
    url: str,
    save_path: str,
    cache: Optional[FileCache] = None,
    progress: Optional[Callable] = None,
    headers: Optional[Dict] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Load image from url to host by target path.

    :param url: Target file path.
    :type url: str
    :param url: The path where the file is saved.
    :type url: str
    :param cache: An instance of FileCache class that provides caching functionality for the downloaded content. If None, caching is disabled.
    :type cache: FileCache, optional
    :param progress: Function for tracking download progress.
    :type progress: Progress, optional
    :param headers: A dictionary of HTTP headers to include in the request.
    :type headers: Dict, optional.
    :param timeout: The maximum number of seconds to wait for a response from the server. If the server does not respond within the timeout period, a TimeoutError is raised.
    :type timeout: int, optional.
    :returns: Full path to downloaded image
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import download
        img_link = 'https://m.media-amazon.com/images/M/MV5BMTYwOTEwNjAzMl5BMl5BanBnXkFtZTcwODc5MTUwMw@@._V1_.jpg'
        im_path = download(img_link, '/home/admin/work/projects/examples/avatar.jpeg')
        print(im_path)
        # Output:
        # /home/admin/work/projects/examples/avatar.jpeg

        # if you need to specify some headers
        headers = {
            'User-Agent': 'Mozilla/5.0',
        }
        im_path = download(img_link, '/home/admin/work/projects/examples/avatar.jpeg', headers=headers)
        print(im_path)
        # Output:
        # /home/admin/work/projects/examples/avatar.jpeg

    """

    def _download():
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
                r.raise_for_status()
                total_size_in_bytes = int(CaseInsensitiveDict(r.headers).get("Content-Length", "0"))
                if progress is not None and type(progress) is Progress:
                    progress.set(0, total_size_in_bytes)
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        if progress is not None:
                            if type(progress) is Progress:
                                progress.iters_done_report(len(chunk))
                            else:
                                progress(len(chunk))
        except requests.exceptions.Timeout as e:
            message = (
                "Request timed out. "
                "This may be due to server-side security measures, network congestion, or other issues. "
                "Please check your server logs for more information."
            )
            logger.warn(msg=message)
            raise e

    if cache is None:
        _download()
    else:
        cache_path = cache.check_storage_object(get_string_hash(url), get_file_ext(save_path))
        if cache_path is None:
            # file not in cache
            _download()
            cache.write_object(save_path, get_string_hash(url))
        else:
            cache.read_object(get_string_hash(url), save_path)
            if progress is not None:
                sizeb = get_file_size(save_path)
                if type(progress) is Progress:
                    progress.set(0, sizeb)
                    progress.iters_done_report(sizeb)
                else:
                    progress(sizeb)

    return save_path


def copy_dir_recursively(
    src_dir: str, dst_dir: str, progress_cb: Optional[Union[tqdm, Callable]] = None
) -> List[str]:
    files = list_files_recursively(src_dir)
    for src_file_path in files:
        dst_file_path = os.path.normpath(src_file_path.replace(src_dir, dst_dir))
        ensure_base_path(dst_file_path)
        if not file_exists(dst_file_path):
            copy_file(src_file_path, dst_file_path)
            if progress_cb is not None:
                progress_cb(get_file_size(src_file_path))


def is_on_agent(remote_path: str):
    if remote_path.startswith("agent://"):
        return True
    else:
        return False


def parse_agent_id_and_path(remote_path: str) -> int:
    if is_on_agent(remote_path) is False:
        raise ValueError("agent path have to starts with 'agent://<agent-id>/'")
    search = re.search("agent://(\d+)(.*)", remote_path)
    agent_id = int(search.group(1))
    path_in_agent_folder = search.group(2)
    if not path_in_agent_folder.startswith("/"):
        path_in_agent_folder = "/" + path_in_agent_folder
    if remote_path.endswith("/") and not path_in_agent_folder.endswith("/"):
        path_in_agent_folder += "/"
    # path_in_agent_folder = os.path.normpath(path_in_agent_folder)
    return agent_id, path_in_agent_folder


def dirs_with_marker(
    input_path: str,
    markers: Union[str, List[str]],
    check_function: Callable = None,
    ignore_case: bool = False,
) -> Generator[str, None, None]:
    """
    Generator that yields paths to directories that contain markers files. If the check_function is specified,
    then the markered directory will be yielded only if the check_function returns True. The check_function
    must take a single argument - the path to the markered directory and return True or False.

    :param input_path: path to the directory in which the search will be performed
    :type input_path: str
    :param markers: single marker or list of markers (e.g. 'config.json' or ['config.json', 'config.yaml'])
    :type markers: Union[str, List[str]]
    :param check_function: function to check that directory meets the requirements and returns bool
    :type check_function: Callable
    :param ignore_case: ignore case when searching for markers
    :type ignore_case: bool
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        input_path = '/home/admin/work/projects/examples'

        # You can pass a string if you have only one marker.
        # markers = 'config.json'

        # Or a list of strings if you have several markers.
        # There's no need to pass one marker in different cases, you can use ignore_case=True for this.
        markers = ['config.json', 'config.yaml']


        # Check function is optional, if you don't need the directories to meet any requirements,
        # you can omit it.

        def check_function(dir_path):
            test_file_path = os.path.join(dir_path, 'test.txt')
            return os.path.exists(test_file_path)

        for directory in sly.fs.dirs_with_marker(input_path, markers, check_function, ignore_case=True):
            # Now you can be sure that the directory contains the markers and meets the requirements.
            # Do something with it.
            print(directory)
    """

    if isinstance(markers, str):
        markers = [markers]

    paths = list_dir_recursively(input_path)
    for path in paths:
        for marker in markers:
            filename = get_file_name_with_ext(path)
            if ignore_case:
                filename = filename.lower()
                marker = marker.lower()

            if filename == marker:
                parent_dir = os.path.dirname(path)
                markered_dir = os.path.join(input_path, parent_dir)

                if check_function is None or check_function(markered_dir):
                    yield markered_dir


def dirs_filter(input_path: str, check_function: Callable) -> Generator[str, None, None]:
    """
    Generator that yields paths to directories that meet the requirements of the check_function.

    :param input_path: path to the directory in which the search will be performed
    :type input_path: str
    :param check_function: function to check that directory meets the requirements and returns bool
    :type check_function: Callable
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        input_path = '/home/admin/work/projects/examples'

        # Prepare the check function.

        def check_function(directory) -> bool:
            images_dir = os.path.join(directory, "images")
            annotations_dir = os.path.join(directory, "annotations")

            return os.path.isdir(images_dir) and os.path.isdir(annotations_dir)

        for directory in sly.fs.dirs(input_path, check_function):
            # Now you can be sure that the directory meets the requirements.
            # Do something with it.
            print(directory)
    """
    paths = [os.path.abspath(input_path)]
    paths.extend(list_dir_recursively(input_path, include_subdirs=True, use_global_paths=True))
    for path in paths:
        if os.path.isdir(path):
            if check_function(path):
                yield path
