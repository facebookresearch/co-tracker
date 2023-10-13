# coding: utf-8
"""download/upload/manipulate files from/to Supervisely team files"""

from __future__ import annotations

from typing import NamedTuple, List, Dict, Optional, Callable, Union
from typing_extensions import Literal

import os
import shutil
import tarfile
from pathlib import Path
import urllib
import re

from supervisely._utils import batched, rand_str
from supervisely.api.module_api import ModuleApiBase, ApiField
import supervisely.io.fs as sly_fs
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import mimetypes
from supervisely.imaging.image import write_bytes, get_hash
from supervisely.task.progress import Progress
from supervisely.io.fs_cache import FileCache
from supervisely.io.fs import (
    get_file_hash,
    get_file_name,
    get_file_ext,
    get_file_size,
    list_files_recursively,
    silent_remove,
    ensure_base_path,
    get_file_name_with_ext,
)
from supervisely.sly_logger import logger
import supervisely.io.env as env
from tqdm import tqdm


class FileInfo(NamedTuple):
    """ """

    team_id: int
    id: int
    user_id: int
    name: str
    hash: str
    path: str
    storage_path: str
    mime: str
    ext: str
    sizeb: int
    created_at: str
    updated_at: str
    full_storage_url: str
    is_dir: bool


class FileApi(ModuleApiBase):
    """
    API for working with Files. :class:`FileApi<FileApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        team_id = 8
        file_path = "/999_App_Test/"
        files = api.file.list(team_id, file_path)
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple FileInfo information about File.

        :Example:

         .. code-block:: python

            FileInfo(team_id=8,
                     id=7660,
                     user_id=7,
                     name='00135.json',
                     hash='z7Hv9a7WIC5HIJrfX/69KVrvtDaLqucSprWHoCxyq0M=',
                     path='/999_App_Test/ds1/00135.json',
                     storage_path='/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json',
                     mime='application/json',
                     ext='json',
                     sizeb=261,
                     created_at='2021-01-11T09:04:17.959Z',
                     updated_at='2021-01-11T09:04:17.959Z',
                     full_storage_url='http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json')
        """
        return [
            ApiField.TEAM_ID,
            ApiField.ID,
            ApiField.USER_ID,
            ApiField.NAME,
            ApiField.HASH,
            ApiField.PATH,
            ApiField.STORAGE_PATH,
            ApiField.MIME2,
            ApiField.EXT2,
            ApiField.SIZEB2,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.FULL_STORAGE_URL,
            ApiField.IS_DIR,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **FileInfo**.
        """
        return "FileInfo"

    def list_on_agent(
        self,
        team_id: int,
        path: str,
        recursive: bool = True,
        return_type: Literal["dict", "fileinfo"] = "dict",
    ) -> List[Union[Dict, FileInfo]]:
        if self.is_on_agent(path) is False:
            raise ValueError(f"Data is not on agent: {path}")

        agent_id, path_in_agent_folder = self.parse_agent_id_and_path(path)
        dirs_queue: List[str] = [path_in_agent_folder]

        results = []
        while len(dirs_queue) > 0:
            cur_dir = dirs_queue.pop(0)
            if cur_dir.endswith("/") is False:
                cur_dir += "/"
            response = self._api.post(
                "agents.storage.list",
                {ApiField.ID: agent_id, ApiField.TEAM_ID: team_id, ApiField.PATH: cur_dir},
            )
            items = response.json()
            for item in items:
                if item["type"] == "file":
                    results.append(item)
                elif item["type"] == "directory" and recursive is True:
                    dirs_queue.append(os.path.join(cur_dir, item["name"]))

        if return_type == "dict":
            return results
        elif return_type == "fileinfo":
            return [self._convert_json_info(info_json) for info_json in results]
        else:
            raise ValueError(
                "The specified value for the 'return_type' parameter should be either 'dict' or 'fileinfo'."
            )

    def list(
        self,
        team_id: int,
        path: str,
        recursive: bool = True,
        return_type: Literal["dict", "fileinfo"] = "dict",
    ) -> List[Union[Dict, FileInfo]]:
        """
        List of files in the Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to File or Directory.
        :type path: str
        :param recursive: If True return all files recursively.
        :type recursive: bool
        :param return_type: The specified value between 'dict' or 'fileinfo'. By default: 'dict'.
        :type return_type: str
        :return: List of all Files with information. See classes info_sequence and FileInfo
        :rtype: class List[Union[Dict, FileInfo]]
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            if sly.is_development():
                load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            # Pass values into the API constructor (optional, not recommended)
            # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

            team_id = 8
            file_path = "/999_App_Test/"

            # Get information about file in dict way..
            files = api.file.list(team_id, file_path)
            file = files[0]
            print(file['id'])
            # Output: 7660

            print(files)
            # Output: [
            #     {
            #         "id":7660,
            #         "userId":7,
            #         "path":"/999_App_Test/ds1/00135.json",
            #         "storagePath":"/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json",
            #         "meta":{
            #             "ext":"json",
            #             "mime":"application/json",
            #             "size":261
            #         },
            #         "createdAt":"2021-01-11T09:04:17.959Z",
            #         "updatedAt":"2021-01-11T09:04:17.959Z",
            #         "hash":"z7Wv1a7WIC5HIJrfX/69XXrqtDaLxucSprWHoCxyq0M=",
            #         "fullStorageUrl":"http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json",
            #         "teamId":8,
            #         "name":"00135.json"
            #     },
            #     {
            #         "id":7661,
            #         "userId":7,
            #         "path":"/999_App_Test/ds1/01587.json",
            #         "storagePath":"/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/9/k/Hs/...json",
            #         "meta":{
            #             "ext":"json",
            #             "mime":"application/json",
            #             "size":252
            #         },
            #         "createdAt":"2021-01-11T09:04:18.099Z",
            #         "updatedAt":"2021-01-11T09:04:18.099Z",
            #         "hash":"La9+XtF2+cTlAqUE/I72e/xS12LqyH1+z<3T+SgD4CTU=",
            #         "fullStorageUrl":"http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/9/k/Hs/...json",
            #         "teamId":8,
            #         "name":"01587.json"
            #     }
            # ]

            # ..or as FileInfo with attributes:
            files = api.file.list(team_id, file_path, return_type='fileinfo')
            file = files[0]
            print(file.id)
            # Output: 7660

            print(files)
            # Output: [
            # FileInfo(team_id=8, id=7660, user_id=7, name='00135.json', hash='z7Wv1a7WI...
            # FileInfo(team_id=8, id=7661, user_id=7, name='01587.json', hash='La9+XtF2+...
            # ]
        """

        if not path.endswith("/") and recursive is False:
            path += "/"
        if self.is_on_agent(path) is True:
            return self.list_on_agent(team_id, path, recursive, return_type)

        response = self._api.post(
            "file-storage.list",
            {ApiField.TEAM_ID: team_id, ApiField.PATH: path, ApiField.RECURSIVE: recursive},
        )

        if return_type == "dict":
            return response.json()
        elif return_type == "fileinfo":
            return [self._convert_json_info(info_json) for info_json in response.json()]
        else:
            raise ValueError(
                "The specified value for the 'return_type' parameter should be either 'dict' or 'fileinfo'."
            )

    def list2(self, team_id: int, path: str, recursive: bool = True) -> List[FileInfo]:
        """
        Disclaimer: Method is not recommended. Use api.file.list instead

        List of files in the Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to File or Directory.
        :type path: str
        :param recursive: If True return all FileInfos recursively.
        :type recursive: bool
        :return: List of all Files with information. See class info_sequence
        :rtype: class List[FileInfo]
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            if sly.is_development():
                load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            # Pass values into the API constructor (optional, not recommended)
            # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

            team_id = 9
            file_path = "/My_App_Test/"
            files = api.file.list2(team_id, file_path)

            print(files)
            # Output: [
            # FileInfo(team_id=9, id=18421, user_id=8, name='5071_3734_mot_video_002.tar.gz', hash='+0nrNoDjBxxJA...
            # FileInfo(team_id=9, id=18456, user_id=8, name='5164_4218_mot_video_bitmap.tar.gz', hash='fwtVI+iptY...
            # FileInfo(team_id=9, id=18453, user_id=8, name='all_vars.tar', hash='TVkUE+K1bnEb9QrdEm9akmHm/QEWPJK...
            # ]
        """
        return self.list(team_id=team_id, path=path, recursive=recursive, return_type="fileinfo")

    def listdir(self, team_id: int, path: str, recursive: bool = False) -> List[str]:
        """
        List dirs and files in the directiry with given path.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to directory.
        :type path: str
        :param recursive: If True return all paths recursively.
        :type recursive: bool
        :return: List of paths
        :rtype: :class:`List[str]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 8
            path = "/999_App_Test/"
            files = api.file.listdir(team_id, path)

            print(files)
            # Output: ["/999_App_Test/ds1", "/999_App_Test/image.png"]
        """
        files = self.list(team_id, path, recursive)
        files_paths = [file["path"] for file in files]
        return files_paths

    def get_directory_size(self, team_id: int, path: str) -> int:
        """
        Get directory size in the Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to Directory.
        :type path: str
        :return: Directory size in the Team Files
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 9
            path = "/My_App_Test/"
            size = api.file.get_directory_size(team_id, path)

            print(size)
            # Output: 3478687
        """
        dir_size = 0
        file_infos = self.list2(team_id, path)
        for file_info in file_infos:
            dir_size += file_info.sizeb
        return dir_size

    def _download(
        self, team_id, remote_path, local_save_path, progress_cb=None
    ):  # TODO: progress bar
        response = self._api.post(
            "file-storage.download",
            {ApiField.TEAM_ID: team_id, ApiField.PATH: remote_path},
            stream=True,
        )
        # print(response.headers)
        # print(response.headers['Content-Length'])
        ensure_base_path(local_save_path)
        with open(local_save_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
                if progress_cb is not None:
                    progress_cb(len(chunk))

    def download(
        self,
        team_id: int,
        remote_path: str,
        local_save_path: str,
        cache: Optional[FileCache] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download File from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Path to File in Team Files.
        :type remote_path: str
        :param local_save_path: Local save path.
        :type local_save_path: str
        :param cache: optional
        :type cache: FileCache, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_file = "/999_App_Test/ds1/01587.json"
            local_save_path = "/home/admin/Downloads/01587.json"

            api.file.download(8, path_to_file, local_save_path)
        """
        if self.is_on_agent(remote_path):
            self.download_from_agent(remote_path, local_save_path, progress_cb)
            return

        if cache is None:
            self._download(team_id, remote_path, local_save_path, progress_cb)
        else:
            file_info = self.get_info_by_path(team_id, remote_path)
            if file_info.hash is None:
                self._download(team_id, remote_path, local_save_path, progress_cb)
            else:
                cache_path = cache.check_storage_object(file_info.hash, get_file_ext(remote_path))
                if cache_path is None:
                    # file not in cache
                    self._download(team_id, remote_path, local_save_path, progress_cb)
                    if file_info.hash != get_file_hash(local_save_path):
                        raise KeyError(
                            f"Remote and local hashes are different (team id: {team_id}, file: {remote_path})"
                        )
                    cache.write_object(local_save_path, file_info.hash)
                else:
                    cache.read_object(file_info.hash, local_save_path)
                    if progress_cb is not None:
                        progress_cb(get_file_size(local_save_path))

    def is_on_agent(self, remote_path: str):
        return sly_fs.is_on_agent(remote_path)

    def parse_agent_id_and_path(self, remote_path: str) -> int:
        return sly_fs.parse_agent_id_and_path(remote_path)

    def download_from_agent(
        self,
        remote_path: str,
        local_save_path: str,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        agent_id, path_in_agent_folder = self.parse_agent_id_and_path(remote_path)
        if (
            agent_id == env.agent_id(raise_not_found=False)
            and env.agent_storage(raise_not_found=False) is not None
        ):
            path_on_agent = os.path.normpath(env.agent_storage() + path_in_agent_folder)
            logger.info(f"Optimized download from agent: {path_on_agent}")
            sly_fs.copy_file(path_on_agent, local_save_path)
            if progress_cb is not None:
                progress_cb(sly_fs.get_file_size(path_on_agent))
            return

        response = self._api.post(
            "agents.storage.download",
            {ApiField.ID: agent_id, ApiField.PATH: path_in_agent_folder},
            stream=True,
        )
        ensure_base_path(local_save_path)
        with open(local_save_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
                if progress_cb is not None:
                    progress_cb(len(chunk))

    def download_directory(
        self,
        team_id: int,
        remote_path: str,
        local_save_path: str,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download Directory from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Path to Directory in Team Files.
        :type remote_path: str
        :param local_save_path: Local save path.
        :type local_save_path: str
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_dir = "/My_App_Test/ds1"
            local_save_path = "/home/admin/Downloads/My_local_test"

            api.file.download_directory(9, path_to_dir, local_save_path)
        """
        if not remote_path.endswith("/"):
            remote_path += "/"

        if self.is_on_agent(remote_path) is True:
            agent_id, path_in_agent_folder = self.parse_agent_id_and_path(remote_path)
            if (
                agent_id == env.agent_id(raise_not_found=False)
                and env.agent_storage(raise_not_found=False) is not None
            ):
                dir_on_agent = os.path.normpath(env.agent_storage() + path_in_agent_folder)
                logger.info(f"Optimized download from agent: {dir_on_agent}")
                sly_fs.copy_dir_recursively(dir_on_agent, local_save_path)
                return

        local_temp_archive = os.path.join(local_save_path, "temp.tar")
        self.download(team_id, remote_path, local_temp_archive, cache=None, progress_cb=progress_cb)
        tr = tarfile.open(local_temp_archive)
        tr.extractall(local_save_path)
        silent_remove(local_temp_archive)
        temp_dir = os.path.join(local_save_path, rand_str(10))
        to_move_dir = os.path.join(local_save_path, os.path.basename(os.path.normpath(remote_path)))
        os.rename(to_move_dir, temp_dir)
        file_names = os.listdir(temp_dir)
        for file_name in file_names:
            shutil.move(os.path.join(temp_dir, file_name), local_save_path)
        shutil.rmtree(temp_dir)

    def _upload_legacy(self, team_id, src, dst):
        """ """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        item = get_file_name_with_ext(dst)
        content_dict = {}
        content_dict[ApiField.NAME] = item

        dst_dir = os.path.dirname(dst)
        if not dst_dir.endswith("/"):
            dst_dir += "/"
        content_dict[ApiField.PATH] = dst_dir  # os.path.basedir ...
        content_dict["file"] = (
            item,
            path_to_bytes_stream(src),
            mimetypes.MimeTypes().guess_type(src)[0],
        )
        encoder = MultipartEncoder(fields=content_dict)
        resp = self._api.post("file-storage.upload?teamId={}".format(team_id), encoder)
        return resp.json()

    def upload(
        self, team_id: int, src: str, dst: str, progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> FileInfo:
        """
        Upload File to Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param src: Local source file path.
        :type src: str
        :param dst: Path to File in Team Files.
        :type dst: str
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about File. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`FileInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_path = "/home/admin/Downloads/01587.json"
            dst_remote_path = "/999_App_Test/ds1/01587.json"

            api.file.upload(8, src_path, dst_remote_path)
        """
        return self.upload_bulk(team_id, [src], [dst], progress_cb)[0]

    def upload_bulk(
        self,
        team_id: int,
        src_paths: List[str],
        dst_paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[FileInfo]:
        """
        Upload Files to Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param src: Local source file paths.
        :type src: List[str]
        :param dst: Destination paths for Files to Team Files.
        :type dst: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about Files. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[FileInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_paths = ["/home/admin/Downloads/01587.json", "/home/admin/Downloads/01588.json","/home/admin/Downloads/01589.json"]
            dst_remote_paths = ["/999_App_Test/ds1/01587.json", "/999_App_Test/ds1/01588.json", "/999_App_Test/ds1/01589.json"]

            api.file.upload_bulk(8, src_paths, dst_remote_paths)
        """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        content_dict = []
        for idx, (src, dst) in enumerate(zip(src_paths, dst_paths)):
            name = get_file_name_with_ext(dst)
            content_dict.append((ApiField.NAME, name))
            dst_dir = os.path.dirname(dst)
            if not dst_dir.endswith("/"):
                dst_dir += "/"
            content_dict.append((ApiField.PATH, dst_dir))
            content_dict.append(
                (
                    "file",
                    (
                        name,
                        path_to_bytes_stream(src),
                        mimetypes.MimeTypes().guess_type(src)[0],
                    ),
                )
            )
        encoder = MultipartEncoder(fields=content_dict)

        # progress = None
        # if progress_logger is not None:
        #     progress = Progress("Uploading", encoder.len, progress_logger, is_size=True)
        #
        # def _print_progress(monitor):
        #     if progress is not None:
        #         progress.set_current_value(monitor.bytes_read)
        #         print(monitor.bytes_read, monitor.len)
        # last_percent = 0
        # def _progress_callback(monitor):
        #     cur_percent = int(monitor.bytes_read * 100.0 / monitor.len)
        #     if cur_percent - last_percent > 10 or cur_percent == 100:
        #         api.task.set_fields(task_id, [{"field": "data.previewProgress", "payload": cur_percent}])
        #     last_percent = cur_percent

        _progress_cb = progress_cb
        if progress_cb is not None and isinstance(progress_cb, tqdm):
            _progress_cb = progress_cb.get_partial()
        if _progress_cb is None:
            data = encoder
        else:
            data = MultipartEncoderMonitor(encoder, _progress_cb)
        resp = self._api.post("file-storage.bulk.upload?teamId={}".format(team_id), data)
        results = [self._convert_json_info(info_json) for info_json in resp.json()]
        return results

    def rename(self, old_name: str, new_name: str) -> None:
        """
        Renames file in Team Files

        :param old_name: Old File name.
        :type old_name: str
        :param new_name: New File name.
        :type new_name: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # NotImplementedError('Method is not supported')
        """
        pass

    def remove_from_agent(self, team_id: int, path: str) -> None:
        raise NotImplementedError()
        agent_id, path_in_agent_folder = self.parse_agent_id_and_path(path)
        if (
            agent_id == env.agent_id(raise_not_found=False)
            and env.agent_storage(raise_not_found=False) is not None
        ):
            # path_on_agent = os.path.normpath(env.agent_storage() + path_in_agent_folder)
            # logger.info(f"Optimized download from agent: {path_on_agent}")
            # sly_fs.copy_file(path_on_agent, local_save_path)
            return

    def remove(self, team_id: int, path: str) -> None:
        """
        Removes a file from the Team Files. If the specified path is a directory,
        the entire directory (including all recursively included files) will be removed.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path in Team Files.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.file.remove(8, "/999_App_Test/ds1/01587.json") # remove file
            api.file.remove(8, "/999_App_Test/ds1/") # remove folder
        """

        if self.is_on_agent(path) is True:
            # self.remove_from_agent(team_id, path)
            logger.warn(
                f"Data '{path}' is on agent. Method does not support agent storage. Remove your data manually on the computer with agent."
            )
            return

        resp = self._api.post(
            "file-storage.remove", {ApiField.TEAM_ID: team_id, ApiField.PATH: path}
        )

    def remove_file(self, team_id: int, path: str) -> None:
        """
        Removes file from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to File in Team Files.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.file.remove_file(8, "/999_App_Test/ds1/01587.json")
        """

        file_info = self.get_info_by_path(team_id, path)

        if file_info is None:
            raise ValueError(
                f"File not found in Team files. Maybe you entered directory? (Path: '{path}')"
            )

        self.remove(team_id, path)

    def remove_dir(self, team_id: int, path: str) -> None:
        """
        Removes folder from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to folder in Team Files.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.file.remove_dir(8, "/999_App_Test/ds1/")
        """

        if not path.endswith("/"):
            raise ValueError("Please add a slash in the end to recognize path as a directory.")

        if not self.dir_exists(team_id, path):
            raise ValueError(f"Folder not found in Team files. (Path: '{path}')")

        self.remove(team_id, path)

    def remove_batch(
        self,
        team_id: int,
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Removes list of files from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param paths: List of paths to Files in Team Files.
        :type paths: List[str]
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            paths_to_del = [
                "/999_App_Test/ds1/01587.json",
                "/999_App_Test/ds1/01588.json",
                "/999_App_Test/ds1/01587.json"
            ]
            api.file.remove(8, paths_to_del)
        """

        for paths_batch in batched(paths, batch_size=100):
            for path in paths_batch:
                if self.is_on_agent(path) is True:
                    logger.warn(
                        f"Data '{path}' is on agent. File skipped. Method does not support agent storage. Remove your data manually on the computer with agent."
                    )
                    paths_batch.remove(path)

            self._api.post(
                "file-storage.bulk.remove", {ApiField.TEAM_ID: team_id, ApiField.PATHS: paths_batch}
            )
            if progress_cb is not None:
                progress_cb(len(paths_batch))

    def exists(self, team_id: int, remote_path: str) -> bool:
        """
        Checks if file exists in Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Remote path to File in Team Files.
        :type remote_path: str
        :return: True if file exists, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file = api.file.exists(8, "/999_App_Test/ds1/02163.json") # True
           file = api.file.exists(8, "/999_App_Test/ds1/01587.json") # False
        """
        path_infos = self.list(team_id, remote_path)
        for info in path_infos:
            if info["path"] == remote_path:
                return True
        return False

    def dir_exists(self, team_id: int, remote_directory: str) -> bool:
        """
        Checks if directory exists in Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Remote path to directory in Team Files.
        :type remote_path: str
        :return: True if directory exists, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file = api.file.dir_exists(8, "/999_App_Test/")   # True
           file = api.file.dir_exists(8, "/10000_App_Test/") # False
        """
        files_infos = self.list(team_id, remote_directory)
        if len(files_infos) > 0:
            return True
        return False

    def get_free_name(self, team_id: int, path: str) -> str:
        """
        Adds suffix to the end of the file name.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Remote path to file in Team Files.
        :type path: str
        :return: New File name with suffix at the end
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file = api.file.get_free_name(8, "/999_App_Test/ds1/02163.json")
           print(file)
           # Output: /999_App_Test/ds1/02163_000.json
        """
        directory = Path(path).parent
        name = get_file_name(path)
        ext = get_file_ext(path)
        res_name = name
        suffix = 0

        def _combine(suffix: int = None):
            res = "{}/{}".format(directory, res_name)
            if suffix is not None:
                res += "_{:03d}".format(suffix)
            if ext:
                res += "{}".format(ext)
            return res

        res_path = _combine()
        while self.exists(team_id, res_path):
            res_path = _combine(suffix)
            suffix += 1
        return res_path

    def get_url(self, file_id: int) -> str:
        """
        Gets URL for the File by ID.

        :param file_id: File ID in Supervisely.
        :type file_id: int
        :return: File URL
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file_id = 7660
           file_url = sly.api.file.get_url(file_id)
           print(file_url)
           # Output: http://supervise.ly/files/7660
        """
        return f"/files/{file_id}"

    def get_info_by_path(self, team_id: int, remote_path: str) -> FileInfo:
        """
        Gets File information by path in Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Remote path to file in Team Files.
        :type remote_path: str
        :return: Information about File. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`FileInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            file_path = "/999_App_Test/ds1/00135.json"
            file_info = api.file.get_info_by_id(8, file_path)
            print(file_info)
            # Output: FileInfo(team_id=8,
            #                  id=7660,
            #                  user_id=7,
            #                  name='00135.json',
            #                  hash='z7Hv9a7WIC5HIJrfX/69KVrvtDaLqucSprWHoCxyq0M=',
            #                  path='/999_App_Test/ds1/00135.json',
            #                  storage_path='/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json',
            #                  mime='application/json',
            #                  ext='json',
            #                  sizeb=261,
            #                  created_at='2021-01-11T09:04:17.959Z',
            #                  updated_at='2021-01-11T09:04:17.959Z',
            #                  full_storage_url='http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json')
        """
        if self.is_on_agent(remote_path) is True:
            path_infos = self.list_on_agent(team_id, os.path.dirname(remote_path), recursive=False)
            for info in path_infos:
                if info["path"] == remote_path:
                    return self._convert_json_info(info)
        else:
            path_infos = self.list(team_id, remote_path)
            for info in path_infos:
                if info["path"] == remote_path:
                    return self._convert_json_info(info)
        return None

    def _convert_json_info(self, info: dict, skip_missing=True) -> FileInfo:
        """ """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        # if res.storage_path is not None:
        #    res = res._replace(full_storage_url=urllib.parse.urljoin(self._api.server_address, res.storage_path))
        return FileInfo(**res._asdict())

    def get_info_by_id(self, id: int) -> FileInfo:
        """
        Gets information about File by ID.

        :param id: File ID in Supervisely.
        :type id: int
        :return: Information about File. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`FileInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            file_id = 7660
            file_info = api.file.get_info_by_id(file_id)
            print(file_info)
            # Output: FileInfo(team_id=8,
            #                  id=7660,
            #                  user_id=7,
            #                  name='00135.json',
            #                  hash='z7Hv9a7WIC5HIJrfX/69KVrvtDaLqucSprWHoCxyq0M=',
            #                  path='/999_App_Test/ds1/00135.json',
            #                  storage_path='/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json',
            #                  mime='application/json',
            #                  ext='json',
            #                  sizeb=261,
            #                  created_at='2021-01-11T09:04:17.959Z',
            #                  updated_at='2021-01-11T09:04:17.959Z',
            #                  full_storage_url='http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json')
        """
        resp = self._api.post("file-storage.info", {ApiField.ID: id})
        return self._convert_json_info(resp.json())

    def get_free_dir_name(self, team_id: int, dir_path: str) -> str:
        """
        Adds suffix to the end of the Directory name.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param dir_path: Path to Directory in Team Files.
        :type dir_path: str
        :return: New Directory name with suffix at the end
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           new_dir_name = api.file.get_free_dir_name(9, "/My_App_Test")
           print(new_dir_name)
           # Output: /My_App_Test_001
        """
        res_dir = dir_path.rstrip("/")
        suffix = 1
        while self.dir_exists(team_id, res_dir):
            res_dir = dir_path.rstrip("/") + f"_{suffix:03d}"
            suffix += 1
        return res_dir

    def upload_directory(
        self,
        team_id: int,
        local_dir: str,
        remote_dir: str,
        change_name_if_conflict: Optional[bool] = True,
        progress_size_cb: Optional[Callable] = None,
        replace_if_conflict: Optional[bool] = False,
    ) -> str:
        """
        Upload Directory to Team Files from local path.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param local_dir: Path to local Directory.
        :type local_dir: str
        :param remote_dir: Path to Directory in Team Files.
        :type remote_dir: str
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :param progress_size_cb: Function for tracking download progress.
        :type progress_size_cb: Progress, optional
        :return: Path to Directory in Team Files
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_dir = "/My_App_Test/ds1"
            local_path = "/home/admin/Downloads/My_local_test"

            api.file.upload_directory(9, local_path, path_to_dir)
        """
        if self.dir_exists(team_id, remote_dir):
            if change_name_if_conflict is True:
                res_remote_dir = self.get_free_dir_name(team_id, remote_dir)
            elif replace_if_conflict is True:
                res_remote_dir = remote_dir
            else:
                raise FileExistsError(
                    f"Directory {remote_dir} already exists in your team (id={team_id})"
                )
        else:
            res_remote_dir = remote_dir

        local_files = list_files_recursively(local_dir)
        remote_files = [
            Path(file.replace(local_dir.rstrip("/"), res_remote_dir.rstrip("/"))).as_posix()
            for file in local_files
        ]

        for local_paths_batch, remote_files_batch in zip(
            batched(local_files, batch_size=50), batched(remote_files, batch_size=50)
        ):
            self.upload_bulk(team_id, local_paths_batch, remote_files_batch, progress_size_cb)
        return res_remote_dir
