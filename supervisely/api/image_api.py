# coding: utf-8
"""Download/upload images from/to Supervisely."""

# docs
from __future__ import annotations

import io
import json
import re
import urllib.parse
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
    Generator,
    Tuple,
    Any,
)

import numpy as np
from requests_toolbelt import MultipartDecoder, MultipartEncoder
from tqdm import tqdm

from supervisely._utils import (
    abs_url,
    batched,
    compress_image_url,
    generate_free_name,
    is_development,
)
from supervisely.annotation.tag_meta import TagMeta
from supervisely.api.module_api import (
    ApiField,
    RemoveableBulkModuleApi,
    _get_single_item,
)
from supervisely.imaging import image as sly_image
from supervisely.io.fs import (
    ensure_base_path,
    get_file_ext,
    get_file_hash,
    get_file_name,
)
from supervisely.sly_logger import logger


class ImageInfo(NamedTuple):
    """
    Object with image parameters from Supervisely.

    :Example:

     .. code-block:: python

        ImageInfo(
            id=770915,
            name='IMG_3861.jpeg',
            link=None,
            hash='ZdpMD+ZMJx0R8BgsCzJcqM7qP4M8f1AEtoYc87xZmyQ=',
            mime='image/jpeg',
            ext='jpeg',
            size=148388,
            width=1067,
            height=800,
            labels_count=4,
            dataset_id=2532,
            created_at='2021-03-02T10:04:33.973Z',
            updated_at='2021-03-02T10:04:33.973Z',
            meta={},
            path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg',
            full_storage_url='http://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg'),
            tags=[]
        )
    """

    #: :class:`int`: Image ID in Supervisely.
    id: int

    #: :class:`str`: Image filename.
    name: str

    #: :class:`str`: Use link as ID for images that are expected to be stored at remote server.
    #: e.g. "http://your-server/image1.jpg".
    link: str

    #: :class:`str`: Image hash obtained by base64(sha256(file_content)).
    #: Use hash for files that are expected to be stored at Supervisely or your deployed agent.
    hash: str

    #: :class:`str`: Image MIME type.
    mime: str

    #: :class:`str`: Image file extension.
    ext: str

    #: :class:`int`: Image size (in bytes).
    size: int

    #: :class:`int`: Image width.
    width: int

    #: :class:`int`: Image height.
    height: int

    #: :class:`int`: Number of :class:`Labels<supervisely.annotation.label.Label>` in the Image.
    labels_count: int

    #: :class:`int`: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
    dataset_id: int

    #: :class:`str`: Image creation time. e.g. "2019-02-22T14:59:53.381Z".
    created_at: str

    #: :class:`str`: Time of last image update. e.g. "2019-02-22T14:59:53.381Z".
    updated_at: str

    #: :class:`dict`: Custom additional image info.
    meta: dict

    #: :class:`str`: Relative storage URL to image. e.g.
    #: "/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg".
    path_original: str

    #: :class:`str`: Full storage URL to image. e.g.
    #: "http://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg".
    full_storage_url: str

    #: :class:`str`: Image :class:`Tags<supervisely.annotation.tag.Tag>` list.
    #: e.g. "[{'entityId': 2836466, 'tagId': 345022, 'id': 2224609, 'labelerLogin': 'admin',
    #: 'createdAt': '2021-03-05T14:15:39.923Z', 'updatedAt': '2021-03-05T14:15:39.923Z'}, {...}]".
    tags: List[Dict]

    @property
    def preview_url(self):
        """
        Get Image preview URL.

        :return: Image preview URL.
        :rtype: :class:`str`
        """
        res = self.full_storage_url
        if is_development():
            res = abs_url(res)
        res = compress_image_url(url=res)
        return res


class ImageApi(RemoveableBulkModuleApi):
    """
    API for working with :class:`Image<supervisely.imaging.image>`. :class:`ImageApi<ImageApi>` object is immutable.

    :param api: API connection to the server
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

        image_info = api.image.get_info_by_id(image_id) # api usage example
    """

    @staticmethod
    def info_sequence():
        """
        Get list of all :class:`ImageInfo<ImageInfo>` field names.

        :return: List of :class:`ImageInfo<ImageInfo>` field names.`
        :rtype: :class:`List[str]`
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.LINK,
            ApiField.HASH,
            ApiField.MIME,
            ApiField.EXT,
            ApiField.SIZE,
            ApiField.WIDTH,
            ApiField.HEIGHT,
            ApiField.LABELS_COUNT,
            ApiField.DATASET_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.META,
            ApiField.PATH_ORIGINAL,
            ApiField.FULL_STORAGE_URL,
            ApiField.TAGS,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of :class:`ImageInfo<ImageInfo>` NamedTuple.

        :return: NamedTuple name.
        :rtype: :class:`str`
        """
        return "ImageInfo"

    def get_list_generator(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        sort: Optional[str] = "id",
        sort_order: Optional[str] = "asc",
        limit: Optional[int] = None,
        force_metadata_for_links: Optional[bool] = False,
        batch_size: Optional[int] = None,
    ) -> Iterator[List[ImageInfo]]:
        data = {
            ApiField.DATASET_ID: dataset_id,
            ApiField.FILTER: filters or [],
            ApiField.SORT: sort,
            ApiField.SORT_ORDER: sort_order,
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            ApiField.PAGINATION_MODE: ApiField.TOKEN,
        }
        if batch_size is not None:
            data[ApiField.PER_PAGE] = batch_size
        else:
            # use default value on instance (20k)
            # #tag/Images/paths/~1images.list/get
            pass
        return self.get_list_all_pages_generator(
            "images.list",
            data,
            limit=limit,
            return_first_response=False,
        )

    def get_list(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        sort: Optional[str] = "id",
        sort_order: Optional[str] = "asc",
        limit: Optional[int] = None,
        force_metadata_for_links: Optional[bool] = True,
        return_first_response: Optional[bool] = False,
    ) -> List[ImageInfo]:
        """
        List of Images in the given :class:`Dataset<supervisely.project.project.Dataset>`.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in which the Images are located.
        :type dataset_id: :class:`int`
        :param filters: List of params to sort output Images.
        :type filters: :class:`List[Dict]`, optional
        :param sort: Field name to sort. One of {'id' (default), 'name', 'description', 'labelsCount', 'createdAt', 'updatedAt'}
        :type sort: :class:`str`, optional
        :param sort_order: Sort order. One of {'asc' (default), 'desc'}
        :type sort_order: :class:`str`, optional
        :param limit: Max number of list elements. No limit if None (default).
        :type limit: :class:`int`, optional
        :return: Objects with image information from Supervisely.
        :rtype: :class:`List[ImageInfo]<ImageInfo>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # Get list of Images with width = 1067
            img_infos = api.image.get_list(dataset_id, filters=[{ 'field': 'width', 'operator': '=', 'value': '1067' }])
            print(img_infos)
            # Output: [ImageInfo(id=770915,
            #                    name='IMG_3861.jpeg',
            #                    link=None,
            #                    hash='ZdpMD+ZMJx0R8BgsCzJcqM7qP4M8f1AEtoYc87xZmyQ=',
            #                    mime='image/jpeg',
            #                    ext='jpeg',
            #                    size=148388,
            #                    width=1067,
            #                    height=800,
            #                    labels_count=4,
            #                    dataset_id=2532,
            #                    created_at='2021-03-02T10:04:33.973Z',
            #                    updated_at='2021-03-02T10:04:33.973Z',
            #                    meta={},
            #                    path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg',
            #                    full_storage_url='http://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg'),
            #                    tags=[],
            # ImageInfo(id=770916,
            #           name='IMG_1836.jpeg',
            #           link=None,
            #           hash='YZKQrZH5C0rBvGGA3p7hjWahz3/pV09u5m30Bz8GeYs=',
            #           mime='image/jpeg',
            #           ext='jpeg',
            #           size=140222,
            #           width=1067,
            #           height=800,
            #           labels_count=3,
            #           dataset_id=2532,
            #           created_at='2021-03-02T10:04:33.973Z',
            #           updated_at='2021-03-02T10:04:33.973Z',
            #           meta={},
            #           path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/C/Y/Hq/...jpg',
            #           full_storage_url='http://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/C/Y/Hq/...jpg'),
            #           tags=[]
            # ]
        """
        return self.get_list_all_pages(
            "images.list",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.FILTER: filters or [],
                ApiField.SORT: sort,
                ApiField.SORT_ORDER: sort_order,
                ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            },
            limit=limit,
            return_first_response=return_first_response,
        )

    def get_filtered_list(
        self,
        dataset_id: int,
        filters: Optional[List[Dict]] = None,
        sort: Optional[str] = "id",
        sort_order: Optional[str] = "asc",
        force_metadata_for_links: Optional[bool] = True,
        limit: Optional[int] = None,
        return_first_response: Optional[bool] = False,
    ) -> List[ImageInfo]:
        """
        List of filtered Images in the given :class:`Dataset<supervisely.project.project.Dataset>`.
        Differs in a more flexible filter format from the get_list() method.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in which the Images are located.
        :type dataset_id: :class:`int`
        :param filters: List of params to sort output Images.
        :type filters: :class:`List[Dict]`, optional
        :param sort: Field name to sort. One of {'id' (default), 'name', 'description', 'labelsCount', 'createdAt', 'updatedAt'}.
        :type sort: :class:`str`, optional
        :param sort_order: Sort order. One of {'asc' (default), 'desc'}
        :type sort_order: :class:`str`, optional
        :return: Objects with image information from Supervisely.
        :rtype: :class:`List[ImageInfo]<ImageInfo>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # Get list of Images with names containing subsequence '2008'
            img_infos = api.image.get_filtered_list(dataset_id, filters=[{ 'type': 'images_filename', 'data': { 'value': '2008' } }])
        """
        if filters is None or not filters:
            return self.get_list(
                dataset_id,
                sort=sort,
                sort_order=sort_order,
                limit=limit,
                force_metadata_for_links=force_metadata_for_links,
                return_first_response=return_first_response,
            )

        if not all(["type" in filter.keys() for filter in filters]):
            raise ValueError("'type' field not found in filter")
        if not all(["data" in filter.keys() for filter in filters]):
            raise ValueError("'data' field not found in filter")

        allowed_filter_types = [
            "images_filename",
            "images_tag",
            "objects_tag",
            "objects_class",
            "objects_annotator",
            "tagged_by_annotator",
            "issues_count",
        ]
        if not all([filter["type"] in allowed_filter_types for filter in filters]):
            raise ValueError(f"'type' field must be one of: {allowed_filter_types}")

        return self.get_list_all_pages(
            "images.list",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.FILTERS: filters,
                ApiField.SORT: sort,
                ApiField.SORT_ORDER: sort_order,
                ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            },
            limit=limit,
            return_first_response=return_first_response,
        )

    def get_info_by_id(self, id: int, force_metadata_for_links=True) -> ImageInfo:
        """
        Get Image information by ID.

        :param id: Image ID in Supervisely.
        :type id: int
        :return: Object with image information from Supervisely.
        :rtype: :class:`ImageInfo<ImageInfo>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # You can get Image ID by listing all images in the Dataset as shown in get_list
            # Or you can open certain image in Supervisely Annotation Tool UI and get last digits of the URL
            img_info = api.image.get_info_by_id(770918)
        """
        return self._get_info_by_id(
            id,
            "images.info",
            fields={ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links},
        )

    def _get_info_by_filters(self, parent_id, filters, force_metadata_for_links):
        """_get_info_by_filters"""
        items = self.get_list(parent_id, filters, force_metadata_for_links=force_metadata_for_links)
        return _get_single_item(items)

    def get_info_by_name(self, dataset_id, name, force_metadata_for_links=True):
        """get_info_by_name"""
        return self._get_info_by_name(
            get_info_by_filters_fn=lambda module_name: self._get_info_by_filters(
                dataset_id, module_name, force_metadata_for_links
            ),
            name=name,
        )

    # @TODO: reimplement to new method images.bulk.info
    def get_info_by_id_batch(
        self,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links=True,
    ) -> List[ImageInfo]:
        """
        Get Images information by ID.

        :param ids: Images IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Function for tracking the progress.
        :type progress_cb: tqdm or callable, optional
        :return: Objects with image information from Supervisely.
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_ids = [376728, 376729, 376730, 376731, 376732, 376733]
            img_infos = image.get_info_by_id_batch(img_ids)
        """
        results = []
        if len(ids) == 0:
            return results
        dataset_id = self.get_info_by_id(ids[0], force_metadata_for_links=False).dataset_id
        for batch in batched(ids):
            filters = [{"field": ApiField.ID, "operator": "in", "value": batch}]
            results.extend(
                self.get_list_all_pages(
                    "images.list",
                    {
                        ApiField.DATASET_ID: dataset_id,
                        ApiField.FILTER: filters,
                        ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                    },
                )
            )
            if progress_cb is not None:
                progress_cb(len(batch))
        temp_map = {info.id: info for info in results}
        ordered_results = [temp_map[id] for id in ids]
        return ordered_results

    def _download(self, id, is_stream=False):
        """
        :param id: int
        :param is_stream: bool
        :return: Response class object contain metadata of image with given id
        """
        response = self._api.post("images.download", {ApiField.ID: id}, stream=is_stream)
        return response

    def download_np(self, id: int, keep_alpha: Optional[bool] = False) -> np.ndarray:
        """
        Download Image with given id in numpy format.

        :param id: Image ID in Supervisely.
        :type id: int
        :param keep_alpha: If True keeps alpha mask for image, otherwise don't.
        :type keep_alpha: bool, optional
        :return: Image in RGB numpy matrix format
        :rtype: :class:`np.ndarray`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_np = api.image.download_np(770918)
        """
        response = self._download(id)
        img = sly_image.read_bytes(response.content, keep_alpha)
        return img

    def download_path(self, id: int, path: str) -> None:
        """
        Downloads Image from Dataset to local path by ID.

        :param id: Image ID in Supervisely.
        :type id: int
        :param path: Local save path for Image.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_info = api.image.get_info_by_id(770918)
            save_path = os.path.join("/home/admin/work/projects/lemons_annotated/ds1/test_imgs/", img_info.name)

            api.image.download_path(770918, save_path)
        """
        response = self._download(id, is_stream=True)
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def download(self, id: int, path: str) -> None:
        """
        Downloads Image from Dataset to local path by ID.

        :param id: Image ID in Supervisely.
        :type id: int
        :param path: Local save path for Image.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_info = api.image.get_info_by_id(770918)
            save_path = os.path.join("/home/admin/work/projects/lemons_annotated/ds1/test_imgs/", img_info.name)

            api.image.download_path(770918, save_path)
        """
        self.download_path(id=id, path=path)

    def _download_batch(
        self, dataset_id: int, ids: List[int], progress_cb: Optional[Union[tqdm, Callable]] = None
    ):
        """
        Get image id and it content from given dataset and list of images ids.
        """
        for batch_ids in batched(ids):
            response = self._api.post(
                "images.bulk.download",
                {ApiField.DATASET_ID: dataset_id, ApiField.IMAGE_IDS: batch_ids},
            )
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                img_id = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])

                if progress_cb is not None:
                    progress_cb(1)
                yield img_id, part

    def download_paths(
        self,
        dataset_id: int,
        ids: List[int],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download Images with given ids and saves them for the given paths.

        :param dataset_id: Dataset ID in Supervisely, where Images are located.
        :type dataset_id: :class:`int`
        :param ids: List of Image IDs in Supervisely.
        :type ids: :class:`List[int]`
        :param paths: Local save paths for Images.
        :type paths: :class:`List[str]`
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`ValueError` if len(ids) != len(paths)
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            local_save_dir = "/home/admin/work/projects/lemons_annotated/ds1/test_imgs"
            save_paths = []
            image_ids = [771755, 771756, 771757, 771758, 771759, 771760]
            img_infos = api.image.get_info_by_id_batch(image_ids)

            p = tqdm(desc="Images downloaded: ", total=len(img_infos))
            for img_info in img_infos:
                save_paths.append(os.path.join(local_save_dir, img_info.name))

            api.image.download_paths(2573, image_ids, save_paths, progress_cb=p)
            # Progress:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 0, "total": 6, "timestamp": "2021-03-15T19:47:15.406Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 1, "total": 6, "timestamp": "2021-03-15T19:47:16.366Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 2, "total": 6, "timestamp": "2021-03-15T19:47:16.367Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 3, "total": 6, "timestamp": "2021-03-15T19:47:16.367Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 4, "total": 6, "timestamp": "2021-03-15T19:47:16.367Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 5, "total": 6, "timestamp": "2021-03-15T19:47:16.368Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 6, "total": 6, "timestamp": "2021-03-15T19:47:16.368Z", "level": "info"}
        """
        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise ValueError('Can not match "ids" and "paths" lists, len(ids) != len(paths)')

        id_to_path = {id: path for id, path in zip(ids, paths)}
        for img_id, resp_part in self._download_batch(dataset_id, ids, progress_cb):
            with open(id_to_path[img_id], "wb") as w:
                w.write(resp_part.content)

    def download_bytes(
        self, dataset_id: int, ids: List[int], progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> List[bytes]:
        """
        Download Images with given IDs from Dataset in Binary format.

        :param dataset_id: Dataset ID in Supervisely, where Images are located.
        :type dataset_id: int
        :param ids: List of Image IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: List of Images in binary format
        :rtype: :class:`List[bytes]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_bytes = api.image.download_bytes(dataset_id, [770918])
            print(img_bytes)
            # Output: [b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\...]
        """
        if len(ids) == 0:
            return []

        id_to_img = {}
        for img_id, resp_part in self._download_batch(dataset_id, ids, progress_cb):
            id_to_img[img_id] = resp_part.content

        return [id_to_img[id] for id in ids]

    def download_nps(
        self,
        dataset_id: int,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        keep_alpha: Optional[bool] = False,
    ) -> List[np.ndarray]:
        """
        Download Images with given IDs in numpy format.

        :param dataset_id: Dataset ID in Supervisely, where Images are located.
        :type dataset_id: int
        :param ids: List of Images IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param keep_alpha: If True keeps alpha mask for Image, otherwise don't.
        :type keep_alpha: bool, optional
        :return: List of Images in RGB numpy matrix format
        :rtype: :class:`List[np.ndarray]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_ids = [770918, 770919, 770920]
            image_nps = api.image.download_nps(dataset_id, image_ids)
        """
        return [
            sly_image.read_bytes(img_bytes, keep_alpha)
            for img_bytes in self.download_bytes(
                dataset_id=dataset_id, ids=ids, progress_cb=progress_cb
            )
        ]

    def download_nps_generator(
        self,
        dataset_id: int,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        keep_alpha: Optional[bool] = False,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        for img_id, img_part in self._download_batch(dataset_id, ids, progress_cb):
            img_bytes = img_part.content
            yield img_id, sly_image.read_bytes(img_bytes, keep_alpha)

    def check_existing_hashes(
        self, hashes: List[str], progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> List[str]:
        """
        Checks existing hashes for Images.

        :param hashes: List of hashes.
        :type hashes: List[str]
        :param progress_cb: Function for tracking progress of checking.
        :type progress_cb: tqdm or callable, optional
        :return: List of existing hashes
        :rtype: :class:`List[str]`
        :Usage example: Checkout detailed example `here <https://app.supervise.ly/explore/notebooks/guide-10-check-existing-images-and-upload-only-the-new-ones-1545/overview>`_ (you must be logged into your Supervisely account)

         .. code-block:: python

            # Helpful method when your uploading was interrupted
            # You can check what images has been successfully uploaded by their hashes and what not
            # And continue uploading the rest of the images from that point

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # Find project
            project = api.project.get_info_by_id(WORKSPACE_ID, PROJECT_ID)

            # Get paths of all images in a directory
            images_paths = sly.fs.list_files('images_to_upload')

            # Calculate hashes for all images paths
            hash_to_image = {}
            images_hashes = []

            for idx, item in enumerate(images_paths):
                item_hash = sly.fs.get_file_hash(item)
                images_hashes.append(item_hash)
                hash_to_image[item_hash] = item

            # Get hashes that are already on server
            remote_hashes = api.image.check_existing_hashes(images_hashes)
            already_uploaded_images = {hh: hash_to_image[hh] for hh in remote_hashes}
        """
        results = []
        if len(hashes) == 0:
            return results
        for hashes_batch in batched(hashes, batch_size=900):
            response = self._api.post("images.internal.hashes.list", hashes_batch)
            results.extend(response.json())

            if progress_cb is not None:
                progress_cb(len(hashes_batch))
        return results

    def check_image_uploaded(self, hash: str) -> bool:
        """
        Checks if Image has been uploaded.

        :param hash: Image hash in Supervisely.
        :type hash: str
        :return: True if Image with given hash exist, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block::

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_check_uploaded = api.image.check_image_uploaded("YZKQrZH5C0rBvGGA3p7hjWahz3/pV09u5m30Bz8GeYs=")
            print(image_check_uploaded)
            # Output: True
        """
        response = self._api.post("images.internal.hashes.list", [hash])
        results = response.json()
        if len(results) == 0:
            return False
        else:
            return True

    def _upload_uniq_images_single_req(self, func_item_to_byte_stream, hashes_items_to_upload):
        """
        Upload images (binary data) to server with single request.
        Expects unique images that aren't exist at server.
        :param func_item_to_byte_stream: converter for "item" to byte stream
        :param hashes_items_to_upload: list of pairs (hash, item)
        :return: list of hashes for successfully uploaded items
        """
        content_dict = {}
        for idx, (_, item) in enumerate(hashes_items_to_upload):
            content_dict["{}-file".format(idx)] = (
                str(idx),
                func_item_to_byte_stream(item),
                "image/*",
            )
        encoder = MultipartEncoder(fields=content_dict)
        resp = self._api.post("images.bulk.upload", encoder)

        resp_list = json.loads(resp.text)
        remote_hashes = [d["hash"] for d in resp_list if "hash" in d]
        if len(remote_hashes) != len(hashes_items_to_upload):
            problem_items = [
                (hsh, item, resp["errors"])
                for (hsh, item), resp in zip(hashes_items_to_upload, resp_list)
                if resp.get("errors")
            ]
            logger.warn(
                "Not all images were uploaded within request.",
                extra={
                    "total_cnt": len(hashes_items_to_upload),
                    "ok_cnt": len(remote_hashes),
                    "items": problem_items,
                },
            )
        return remote_hashes

    def _upload_data_bulk(
        self, func_item_to_byte_stream, items_hashes, retry_cnt=3, progress_cb=None
    ):
        """
        Upload images (binary data) to server. Works with already existing or duplicating images.
        :param func_item_to_byte_stream: converter for "item" to byte stream
        :param items_hashes: iterable of pairs (item, hash) where "item" is a some descriptor (e.g. image file path)
         for image data, and "hash" is a hash for the image binary data
        :param retry_cnt: int, number of retries to send the whole set of items
        :param progress_cb: callback or tqdm object to account progress (in number of items)
        """
        hash_to_items = {i_hash: item for item, i_hash in items_hashes}

        unique_hashes = set(hash_to_items.keys())
        remote_hashes = set(
            self.check_existing_hashes(list(unique_hashes))
        )  # existing -- from server
        if progress_cb:
            progress_cb(len(remote_hashes))
        pending_hashes = unique_hashes - remote_hashes

        # @TODO: some correlation with sly.io.network_exceptions. Should we perform retries here?
        for retry_idx in range(retry_cnt):
            # single attempt to upload all data which is not uploaded yet

            for hashes in batched(list(pending_hashes)):
                pending_hashes_items = [(h, hash_to_items[h]) for h in hashes]
                hashes_rcv = self._upload_uniq_images_single_req(
                    func_item_to_byte_stream, pending_hashes_items
                )
                pending_hashes -= set(hashes_rcv)
                if set(hashes_rcv) - set(hashes):
                    logger.warn(
                        "Hash inconsistency in images bulk upload.",
                        extra={"sent": hashes, "received": hashes_rcv},
                    )
                if progress_cb:
                    progress_cb(len(hashes_rcv))

            if not pending_hashes:
                return

            warning_items = []
            for h in pending_hashes:
                item_data = hash_to_items[h]
                if isinstance(item_data, (bytes, bytearray)):
                    item_data = "some bytes ..."
                warning_items.append((h, item_data))

            logger.warn(
                "Unable to upload images (data).",
                extra={
                    "retry_idx": retry_idx,
                    "items": warning_items,
                },
            )
            # now retry it for the case if it is a shadow server/connection error

        raise ValueError(
            "Unable to upload images (data). "
            "Please check if images are in supported format and if ones aren't corrupted."
        )

    def upload_path(
        self, dataset_id: int, name: str, path: str, meta: Optional[Dict] = None
    ) -> ImageInfo:
        """
        Uploads Image with given name from given local path to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name.
        :type name: str
        :param path: Local Image path.
        :type path: str
        :param meta: Image metadata.
        :type meta: dict, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_info = api.image.upload_path(dataset_id, name="7777.jpeg", path="/home/admin/Downloads/7777.jpeg")
        """
        metas = None if meta is None else [meta]
        return self.upload_paths(dataset_id, [name], [path], metas=metas)[0]

    def upload_paths(
        self,
        dataset_id: int,
        names: List[str],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
    ) -> List[ImageInfo]:
        """
        Uploads Images with given names from given local path to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of Images names.
        :type names: List[str]
        :param paths: List of local Images pathes.
        :type paths: List[str]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Images metadata.
        :type metas: List[dict], optional
        :raises: :class:`ValueError` if len(names) != len(paths)
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_names = ["7777.jpeg", "8888.jpeg", "9999.jpeg"]
            image_paths = ["/home/admin/Downloads/img/770918.jpeg", "/home/admin/Downloads/img/770919.jpeg", "/home/admin/Downloads/img/770920.jpeg"]

            img_infos = api.image.upload_path(dataset_id, names=img_names, paths=img_paths)
        """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        hashes = [get_file_hash(x) for x in paths]

        self._upload_data_bulk(path_to_bytes_stream, zip(paths, hashes), progress_cb=progress_cb)
        return self.upload_hashes(dataset_id, names, hashes, metas=metas)

    def upload_np(
        self, dataset_id: int, name: str, img: np.ndarray, meta: Optional[Dict] = None
    ) -> ImageInfo:
        """
        Upload given Image in numpy format with given name to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name with extension.
        :type name: str
        :param img: image in RGB format(numpy matrix)
        :type img: np.ndarray
        :param meta: Image metadata.
        :type meta: dict, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_np = sly.image.read("/home/admin/Downloads/7777.jpeg")
            img_info = api.image.upload_np(dataset_id, name="7777.jpeg", img=img_np)
        """
        metas = None if meta is None else [meta]
        return self.upload_nps(dataset_id, [name], [img], metas=metas)[0]

    def upload_nps(
        self,
        dataset_id: int,
        names: List[str],
        imgs: List[np.ndarray],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
    ) -> List[ImageInfo]:
        """
        Upload given Images in numpy format with given names to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Images names.
        :type names: List[str]
        :param imgs: Images in RGB numpy matrix format
        :type imgs: List[np.ndarray]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Images metadata.
        :type metas: List[dict], optional
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_np_1 = sly.image.read("/home/admin/Downloads/7777.jpeg")
            img_np_2 = sly.image.read("/home/admin/Downloads/8888.jpeg")
            img_np_3 = sly.image.read("/home/admin/Downloads/9999.jpeg")

            img_names = ["7777.jpeg", "8888.jpeg", "9999.jpeg"]
            img_nps = [img_np_1, img_np_2, img_np_3]

            img_infos = api.image.upload_nps(dataset_id, names=img_names, imgs=img_nps)
        """

        def img_to_bytes_stream(item):
            img, name = item[0], item[1]
            img_bytes = sly_image.write_bytes(img, get_file_ext(name))
            return io.BytesIO(img_bytes)

        def img_to_hash(item):
            img, name = item[0], item[1]
            return sly_image.get_hash(img, get_file_ext(name))

        img_name_list = list(zip(imgs, names))
        hashes = [img_to_hash(x) for x in img_name_list]

        self._upload_data_bulk(
            img_to_bytes_stream, zip(img_name_list, hashes), progress_cb=progress_cb
        )
        return self.upload_hashes(dataset_id, names, hashes, metas=metas)

    def upload_link(
        self,
        dataset_id: int,
        name: str,
        link: str,
        meta: Optional[Dict] = None,
        force_metadata_for_links=True,
    ) -> ImageInfo:
        """
        Uploads Image from given link to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name.
        :type name: str
        :param link: Link to Image.
        :type link: str
        :param meta: Image metadata.
        :type meta: dict, optional
        :param force_metadata_for_links: Calculate metadata for link. If False, metadata will be empty.
        :type force_metadata_for_links: bool, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_name = 'Avatar.jpg'
            img_link = 'https://m.media-amazon.com/images/M/MV5BMTYwOTEwNjAzMl5BMl5BanBnXkFtZTcwODc5MTUwMw@@._V1_.jpg'

            img_info = api.image.upload_link(dataset_id, img_name, img_link)
        """
        metas = None if meta is None else [meta]
        return self.upload_links(
            dataset_id,
            [name],
            [link],
            metas=metas,
            force_metadata_for_links=force_metadata_for_links,
        )[0]

    def upload_links(
        self,
        dataset_id: int,
        names: List[str],
        links: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        batch_size: Optional[int] = 50,
        force_metadata_for_links: Optional[bool] = True,
        skip_validation: Optional[bool] = False,
    ) -> List[ImageInfo]:
        """
        Uploads Images from given links to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Images names.
        :type names: List[str]
        :param links: Links to Images.
        :type links: List[str]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Images metadata.
        :type metas: List[dict], optional
        :param force_metadata_for_links: Calculate metadata for links. If False, metadata will be empty.
        :type force_metadata_for_links: bool, optional

        :param skip_validation: Skips validation for images, can result in invalid images being uploaded.
        :type skip_validation: bool, optional
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_names = ['Avatar.jpg', 'Harry Potter.jpg', 'Avengers.jpg']
            img_links = ['https://m.media-amazon.com/images/M/MV5BMTYwOTEwNjAzMl5BMl5BanBnXkFtZTcwODc5MTUwMw@@._V1_.jpg',
                         'https://m.media-amazon.com/images/M/MV5BNDYxNjQyMjAtNTdiOS00NGYwLWFmNTAtNThmYjU5ZGI2YTI1XkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_.jpg',
                         'https://m.media-amazon.com/images/M/MV5BNjQ3NWNlNmQtMTE5ZS00MDdmLTlkZjUtZTBlM2UxMGFiMTU3XkEyXkFqcGdeQXVyNjUwNzk3NDc@._V1_.jpg']

            img_infos = api.image.upload_links(dataset_id, img_names, img_links)
        """
        return self._upload_bulk_add(
            lambda item: (ApiField.LINK, item),
            dataset_id,
            names,
            links,
            progress_cb,
            metas=metas,
            batch_size=batch_size,
            force_metadata_for_links=force_metadata_for_links,
            skip_validation=skip_validation,
        )

    def upload_hash(
        self, dataset_id: int, name: str, hash: str, meta: Optional[Dict] = None
    ) -> ImageInfo:
        """
        Upload Image from given hash to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name.
        :type name: str
        :param hash: Image hash.
        :type hash: str
        :param meta: Image metadata.
        :type meta: dict, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_dataset_id = 452984
            im_info = api.image.get_info_by_id(193940090)
            hash = im_info.hash
            # It is necessary to upload image with the same name(extention) as in src dataset
            name = im_info.name
            meta = {1: 'meta_example'}
            new_in_info = api.image.upload_hash(dst_dataset_id, name, hash, meta)
            print(json.dumps(new_in_info, indent=4))
            # Output: [
            #     196793586,
            #     "IMG_0748.jpeg",
            #     null,
            #     "NEjmnmdd7DOzaFAKK/nCIl5CtcwZeMkhW3CHe875p9g=",
            #     "image/jpeg",
            #     "jpeg",
            #     66885,
            #     600,
            #     500,
            #     0,
            #     452984,
            #     "2021-03-16T09:09:45.587Z",
            #     "2021-03-16T09:09:45.587Z",
            #     {
            #         "1": "meta_example"
            #     },
            #     "/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/a/kn/W2mzMQg435d6wG0.jpg",
            #     "https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/a/kn/W2mzMQg435hiHJAPgMU.jpg"
            # ]
        """
        metas = None if meta is None else [meta]
        return self.upload_hashes(dataset_id, [name], [hash], metas=metas)[0]

    def upload_hashes(
        self,
        dataset_id: int,
        names: List[str],
        hashes: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        batch_size: Optional[int] = 50,
        skip_validation: Optional[bool] = False,
    ) -> List[ImageInfo]:
        """
        Upload images from given hashes to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Images names.
        :type names: List[str]
        :param hashes: Images hashes.
        :type hashes: List[str]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Images metadata.
        :type metas: List[dict], optional
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_dataset_id = 447130
            hashes = []
            names = []
            metas = []
            imgs_info = api.image.get_list(src_dataset_id)
            # Create lists of hashes, images names and meta information for each image
            for im_info in imgs_info:
                hashes.append(im_info.hash)
                # It is necessary to upload images with the same names(extentions) as in src dataset
                names.append(im_info.name)
                metas.append({im_info.name: im_info.size})

            dst_dataset_id = 452984
            progress = sly.Progress("Images upload: ", len(hashes))
            new_imgs_info = api.image.upload_hashes(dst_dataset_id, names, hashes, progress.iters_done_report, metas)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 0, "total": 10, "timestamp": "2021-03-16T11:59:07.444Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 10, "total": 10, "timestamp": "2021-03-16T11:59:07.644Z", "level": "info"}
        """
        return self._upload_bulk_add(
            lambda item: (ApiField.HASH, item),
            dataset_id,
            names,
            hashes,
            progress_cb,
            metas=metas,
            batch_size=batch_size,
            skip_validation=skip_validation,
        )

    def upload_id(
        self, dataset_id: int, name: str, id: int, meta: Optional[Dict] = None
    ) -> ImageInfo:
        """
        Upload Image by ID to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name.
        :type name: str
        :param id: Image ID in Supervisely.
        :type id: int
        :param meta: Image metadata.
        :type meta: dict, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_dataset_id = 452984
            im_info = api.image.get_info_by_id(193940090)
            id = im_info.id
            # It is necessary to upload image with the same name(extention) as in src dataset
            name = im_info.name
            meta = {1: 'meta_example'}
            new_in_info = api.image.upload_id(dst_dataset_id, name, id, meta)
            print(json.dumps(new_in_info, indent=4))
            # Output: [
            #     196793605,
            #     "IMG_0748.jpeg",
            #     null,
            #     "NEjmnmdd7DOzaFAKK/nCIl5CtcwZeMkhW3CHe875p9g=",
            #     "image/jpeg",
            #     "jpeg",
            #     66885,
            #     600,
            #     500,
            #     0,
            #     452984,
            #     "2021-03-16T09:27:12.620Z",
            #     "2021-03-16T09:27:12.620Z",
            #     {
            #         "1": "meta_example"
            #     },
            #     "/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/a/kn/W2mzMQg435d6wG0AJGJTOsL1FqMUNOPqu4VdzFAN36LqtGwBIE4AmLOQ1BAxuIyB0bHJAPgMU.jpg",
            #     "https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/a/kn/iEaDEkejnfnb1Tz56ka0hiHJAPgMU.jpg"
            # ]
        """
        metas = None if meta is None else [meta]
        return self.upload_ids(dataset_id, [name], [id], metas=metas)[0]

    def upload_ids(
        self,
        dataset_id: int,
        names: List[str],
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        batch_size: Optional[int] = 50,
        force_metadata_for_links: bool = True,
        infos: List[ImageInfo] = None,
        skip_validation: Optional[bool] = False,
    ) -> List[ImageInfo]:
        """
        Upload Images by IDs to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Images names.
        :type names: List[str]
        :param ids: Images IDs.
        :type ids: List[int]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Images metadata.
        :type metas: List[dict], optional
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_dataset_id = 447130

            ids = []
            names = []
            metas = []
            imgs_info = api.image.get_list(src_dataset_id)
            # Create lists of ids, images names and meta information for each image
            for im_info in imgs_info:
                ids.append(im_info.id)
                # It is necessary to upload images with the same names(extentions) as in src dataset
                names.append(im_info.name)
                metas.append({im_info.name: im_info.size})

            dst_dataset_id = 452984
            progress = sly.Progress("Images upload: ", len(ids))
            new_imgs_info = api.image.upload_ids(dst_dataset_id, names, ids, progress.iters_done_report, metas)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 0, "total": 10, "timestamp": "2021-03-16T12:31:36.550Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 10, "total": 10, "timestamp": "2021-03-16T12:31:37.119Z", "level": "info"}
        """
        if metas is None:
            metas = [{}] * len(names)

        if infos is None:
            infos = self.get_info_by_id_batch(
                ids, force_metadata_for_links=force_metadata_for_links
            )

        # prev implementation
        # hashes = [info.hash for info in infos]
        # return self.upload_hashes(dataset_id, names, hashes, progress_cb, metas=metas)

        links, links_names, links_order, links_metas = [], [], [], []
        hashes, hashes_names, hashes_order, hashes_metas = [], [], [], []
        for idx, (name, info, meta) in enumerate(zip(names, infos, metas)):
            if info.link is not None:
                links.append(info.link)
                links_names.append(name)
                links_order.append(idx)
                links_metas.append(meta)
            else:
                hashes.append(info.hash)
                hashes_names.append(name)
                hashes_order.append(idx)
                hashes_metas.append(meta)

        result = [None] * len(names)
        if len(links) > 0:
            res_infos_links = self.upload_links(
                dataset_id,
                links_names,
                links,
                progress_cb,
                metas=links_metas,
                batch_size=batch_size,
                force_metadata_for_links=force_metadata_for_links,
                skip_validation=skip_validation,
            )
            for info, pos in zip(res_infos_links, links_order):
                result[pos] = info

        if len(hashes) > 0:
            res_infos_hashes = self.upload_hashes(
                dataset_id,
                hashes_names,
                hashes,
                progress_cb,
                metas=hashes_metas,
                batch_size=batch_size,
                skip_validation=skip_validation,
            )
            for info, pos in zip(res_infos_hashes, hashes_order):
                result[pos] = info

        return result

    def _upload_bulk_add(
        self,
        func_item_to_kv,
        dataset_id,
        names,
        items,
        progress_cb=None,
        metas=None,
        batch_size=50,
        force_metadata_for_links=True,
        skip_validation=False,
    ):
        """ """
        results = []

        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise ValueError('Can not match "names" and "items" lists, len(names) != len(items)')

        if metas is None:
            metas = [{}] * len(names)
        else:
            if len(names) != len(metas):
                raise ValueError('Can not match "names" and "metas" len(names) != len(metas)')

        for batch in batched(list(zip(names, items, metas)), batch_size=batch_size):
            images = []
            for name, item, meta in batch:
                item_tuple = func_item_to_kv(item)
                # @TODO: 'title' -> ApiField.NAME
                image_data = {"title": name, item_tuple[0]: item_tuple[1]}
                if len(meta) != 0 and type(meta) == dict:
                    image_data[ApiField.META] = meta
                images.append(image_data)

            response = self._api.post(
                "images.bulk.add",
                {
                    ApiField.DATASET_ID: dataset_id,
                    ApiField.IMAGES: images,
                    ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                    ApiField.SKIP_VALIDATION: skip_validation,
                },
            )
            if progress_cb is not None:
                progress_cb(len(images))

            for info_json in response.json():
                info_json_copy = info_json.copy()
                if info_json.get(ApiField.MIME, None) is not None:
                    info_json_copy[ApiField.EXT] = info_json[ApiField.MIME].split("/")[1]
                # results.append(self.InfoType(*[info_json_copy[field_name] for field_name in self.info_sequence()]))
                results.append(self._convert_json_info(info_json_copy))

        # name_to_res = {img_info.name: img_info for img_info in results}
        # ordered_results = [name_to_res[name] for name in names]

        return results  # ordered_results

    # @TODO: reimplement
    def _convert_json_info(self, info: dict, skip_missing=True):
        """ """
        if info is None:
            return None
        temp_ext = None
        field_values = []
        for field_name in self.info_sequence():
            if field_name == ApiField.EXT:
                continue
            if skip_missing is True:
                val = info.get(field_name, None)
            else:
                val = info[field_name]
            field_values.append(val)
            if field_name == ApiField.MIME:
                if val is not None:
                    temp_ext = val.split("/")[1]
                else:
                    temp_ext = None
                field_values.append(temp_ext)
        for idx, field_name in enumerate(self.info_sequence()):
            if field_name == ApiField.NAME:
                cur_ext = get_file_ext(field_values[idx]).replace(".", "").lower()
                if not cur_ext:
                    field_values[idx] = "{}.{}".format(field_values[idx], temp_ext)
                    break
                if temp_ext is None:
                    break
                if temp_ext == "jpeg" and cur_ext in ["jpg", "jpeg", "mpo"]:
                    break
                if temp_ext != cur_ext:
                    field_values[idx] = "{}.{}".format(field_values[idx], temp_ext)
                break

        res = self.InfoType(*field_values)
        d = res._asdict()

        return ImageInfo(**d)

    def _remove_batch_api_method_name(self):
        """ """
        return "images.bulk.remove"

    def _remove_batch_field_name(self):
        """ """
        return ApiField.IMAGE_IDS

    def copy_batch(
        self,
        dst_dataset_id: int,
        ids: List[int],
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[ImageInfo]:
        """
        Copies Images with given IDs to Dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param ids: Images IDs in Supervisely.
        :type ids: List[int]
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :param progress_cb: Function for tracking the progress of copying.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`TypeError` if type of ids is not list
        :raises: :class:`ValueError` if images ids are from the destination Dataset
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ds_lemon_id = 1780

            ds_lemon_img_infos = api.image.get_list(ds_lemon_id)

            lemons_img_ids = [lemon_img_info.id for lemon_img_info in ds_lemon_img_infos]

            ds_fruit_id = 2574
            ds_fruit_img_infos = api.image.copy_batch(ds_fruit_id, lemons_img_ids, with_annotations=True)
        """
        if type(ids) is not list:
            raise TypeError(
                "ids parameter has type {!r}. but has to be of type {!r}".format(type(ids), list)
            )

        if len(ids) == 0:
            return

        existing_images = self.get_list(dst_dataset_id)
        existing_names = {image.name for image in existing_images}

        ids_info = self.get_info_by_id_batch(ids, force_metadata_for_links=False)
        temp_ds_ids = []
        for info in ids_info:
            if info.dataset_id not in temp_ds_ids:
                temp_ds_ids.append(info.dataset_id)
        if len(temp_ds_ids) > 1:
            raise ValueError("Images ids have to be from the same dataset")

        if change_name_if_conflict:
            new_names = [
                generate_free_name(existing_names, info.name, with_ext=True, extend_used_names=True)
                for info in ids_info
            ]
        else:
            new_names = [info.name for info in ids_info]
            names_intersection = existing_names.intersection(set(new_names))
            if len(names_intersection) != 0:
                raise ValueError(
                    "Images with the same names already exist in destination dataset. "
                    'Please, use argument "change_name_if_conflict=True" to automatically resolve '
                    "names intersection"
                )

        new_images = self.upload_ids(dst_dataset_id, new_names, ids, progress_cb)
        new_ids = [new_image.id for new_image in new_images]

        if with_annotations:
            src_project_id = self._api.dataset.get_info_by_id(ids_info[0].dataset_id).project_id
            dst_project_id = self._api.dataset.get_info_by_id(dst_dataset_id).project_id
            self._api.project.merge_metas(src_project_id, dst_project_id)
            self._api.annotation.copy_batch(ids, new_ids)

        return new_images

    def copy_batch_optimized(
        self,
        src_dataset_id: int,
        src_image_infos: List[ImageInfo],
        dst_dataset_id: int,
        with_annotations: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        dst_names: List[ImageInfo] = None,
        batch_size: Optional[int] = 500,
        skip_validation: Optional[bool] = False,
        save_source_date: Optional[bool] = True,
    ) -> List[ImageInfo]:
        """
        Copies Images with given IDs to Dataset.

        :param src_dataset_id: Source Dataset ID in Supervisely.
        :type src_dataset_id: int
        :param src_image_infos: ImageInfo objects of images to copy.
        :type src_image_infos: List [ :class:`ImageInfo` ]
        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :param progress_cb: Function for tracking the progress of copying.
        :type progress_cb: tqdm or callable, optional
        :param dst_names: ImageInfo list with existing items in destination dataset.
        :type dst_names: List [ :class:`ImageInfo` ], optional
        :param batch_size: Number of elements to copy for each request.
        :type batch_size: int, optional
        :param skip_validation: Flag for skipping additinal validations.
        :type skip_validation: bool, optional
        :param save_source_date: Save source annotation dates (creation and modification) or create a new date.
        :type save_source_date: bool, optional
        :raises: :class:`TypeError` if type of src_image_infos is not list
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_ds_id = 2231
            img_infos = api.image.get_list(src_ds_id)

            dest_ds_id = 2574
            dest_img_infos = api.image.copy_batch_optimized(src_ds_id, img_infos, dest_ds_id)
        """
        if type(src_image_infos) is not list:
            raise TypeError(
                "src_image_infos parameter has type {!r}. but has to be of type {!r}".format(
                    type(src_image_infos), list
                )
            )

        if len(src_image_infos) == 0:
            return

        if dst_names is None:
            existing_images = self.get_list(dst_dataset_id, force_metadata_for_links=False)
            existing_names = {image.name for image in existing_images}
            new_names = [
                generate_free_name(existing_names, info.name, with_ext=True, extend_used_names=True)
                for info in src_image_infos
            ]
        else:
            if len(dst_names) != len(src_image_infos):
                raise RuntimeError("len(dst_names) != len(src_image_infos)")
            new_names = dst_names

        src_ids = [info.id for info in src_image_infos]
        new_images = self.upload_ids(
            dst_dataset_id,
            new_names,
            src_ids,
            progress_cb,
            batch_size=batch_size,
            force_metadata_for_links=False,
            infos=src_image_infos,
            skip_validation=skip_validation,
        )
        new_ids = [new_image.id for new_image in new_images]

        if with_annotations:
            src_project_id = self._api.dataset.get_info_by_id(src_dataset_id).project_id
            dst_project_id = self._api.dataset.get_info_by_id(dst_dataset_id).project_id
            self._api.project.merge_metas(src_project_id, dst_project_id)
            self._api.annotation.copy_batch_by_ids(
                src_ids,
                new_ids,
                batch_size=batch_size,
                save_source_date=save_source_date,
            )

        return new_images

    def move_batch(
        self,
        dst_dataset_id: int,
        ids: List[int],
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[ImageInfo]:
        """
        Moves Images with given IDs to Dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param ids: Images IDs in Supervisely.
        :type ids: List[int]
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :param progress_cb: Function for tracking the progress of moving.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`TypeError` if type of ids is not list
        :raises: :class:`ValueError` if images ids are from the destination Dataset
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ds_lemon_id = 1780
            ds_kiwi_id = 1233

            ds_lemon_img_infos = api.image.get_list(ds_lemon_id)
            ds_kiwi_img_infos = api.image.get_list(ds_kiwi_id)

            fruit_img_ids = []
            for lemon_img_info, kiwi_img_info in zip(ds_lemon_img_infos, ds_kiwi_img_infos):
                fruit_img_ids.append(lemon_img_info.id)
                fruit_img_ids.append(kiwi_img_info.id)

            ds_fruit_id = 2574
            ds_fruit_img_infos = api.image.move_batch(ds_fruit_id, fruit_img_ids, with_annotations=True)
        """
        new_images = self.copy_batch(
            dst_dataset_id, ids, change_name_if_conflict, with_annotations, progress_cb
        )
        self.remove_batch(ids)
        return new_images

    def move_batch_optimized(
        self,
        src_dataset_id: int,
        src_image_infos: List[ImageInfo],
        dst_dataset_id: int,
        with_annotations: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        dst_names: List[ImageInfo] = None,
        batch_size: Optional[int] = 500,
        skip_validation: Optional[bool] = False,
        save_source_date: Optional[bool] = True,
    ) -> List[ImageInfo]:
        """
        Moves Images with given IDs to Dataset.

        :param src_dataset_id: Source Dataset ID in Supervisely.
        :type src_dataset_id: int
        :param src_image_infos: ImageInfo objects of images to move.
        :type src_image_infos: List [ :class:`ImageInfo` ]
        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :param progress_cb: Function for tracking the progress of moving.
        :type progress_cb: tqdm or callable, optional
        :param dst_names: ImageInfo list with existing items in destination dataset.
        :type dst_names: List [ :class:`ImageInfo` ], optional
        :param batch_size: Number of elements to copy for each request.
        :type batch_size: int, optional
        :param skip_validation: Flag for skipping additinal validations.
        :type skip_validation: bool, optional
        :param save_source_date: Save source annotation dates (creation and modification) or create a new date.
        :type save_source_date: bool, optional
        :raises: :class:`TypeError` if type of src_image_infos is not list
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_ds_id = 2231
            img_infos = api.image.get_list(src_ds_id)

            dest_ds_id = 2574
            dest_img_infos = api.image.move_batch_optimized(src_ds_id, img_infos, dest_ds_id)
        """
        new_images = self.copy_batch_optimized(
            src_dataset_id,
            src_image_infos,
            dst_dataset_id,
            with_annotations=with_annotations,
            progress_cb=progress_cb,
            dst_names=dst_names,
            batch_size=batch_size,
            skip_validation=skip_validation,
            save_source_date=save_source_date,
        )
        src_ids = [info.id for info in src_image_infos]
        self.remove_batch(src_ids, batch_size=batch_size)
        return new_images

    def copy(
        self,
        dst_dataset_id: int,
        id: int,
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
    ) -> ImageInfo:
        """
        Copies Image with given ID to destination Dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param id: Image ID in Supervisely.
        :type id: int
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_ds_id = 365184
            img_id = 121236920

            img_info = api.image.copy(dst_ds_id, img_id, with_annotations=True)
        """
        return self.copy_batch(dst_dataset_id, [id], change_name_if_conflict, with_annotations)[0]

    def move(
        self,
        dst_dataset_id: int,
        id: int,
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
    ) -> ImageInfo:
        """
        Moves Image with given ID to destination Dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param id: Image ID in Supervisely.
        :type id: int
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_ds_id = 365484
            img_id = 533336920

            img_info = api.image.copy(dst_ds_id, img_id, with_annotations=True)
        """
        return self.move_batch(dst_dataset_id, [id], change_name_if_conflict, with_annotations)[0]

    def url(
        self,
        team_id: int,
        workspace_id: int,
        project_id: int,
        dataset_id: int,
        image_id: int,
    ) -> str:
        """
        Gets Image URL by ID.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param workspace_id: Workspace ID in Supervisely.
        :type workspace_id: int
        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :return: Image URL
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 16087
            workspace_id = 23821
            project_id = 53939
            dataset_id = 254737
            image_id = 121236920

            img_url = api.image.url(team_id, workspace_id, project_id, dataset_id, image_id)
            print(url)
            # Output: https://app.supervise.ly/app/images/16087/23821/53939/254737#image-121236920
        """
        result = urllib.parse.urljoin(
            self._api.server_address,
            "app/images/{}/{}/{}/{}#image-{}".format(
                team_id, workspace_id, project_id, dataset_id, image_id
            ),
        )

        return result

    def _download_batch_by_hashes(self, hashes) -> Generator[Tuple[str, Any], None, None]:
        """ """
        for batch_hashes in batched(hashes):
            response = self._api.post(
                "images.bulk.download-by-hash", {ApiField.HASHES: batch_hashes}
            )
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                h = content_utf8.replace('form-data; name="', "")[:-1]
                yield h, part

    def download_paths_by_hashes(
        self,
        hashes: List[str],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download Images with given hashes in Supervisely server and saves them for the given paths.

        :param hashes: List of images hashes in Supervisely.
        :type hashes: List[str]
        :param paths: List of paths to save images.
        :type paths: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`ValueError` if len(hashes) != len(paths)
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 447130
            dir_for_save = '/home/admin/Downloads/img'
            hashes = []
            paths = []
            imgs_info = api.image.get_list(dataset_id)
            for im_info in imgs_info:
                hashes.append(im_info.hash)
                # It is necessary to save images with the same names(extentions) as on the server
                paths.append(os.path.join(dir_for_save, im_info.name))
            api.image.download_paths_by_hashes(hashes, paths)
        """
        if len(hashes) == 0:
            return
        if len(hashes) != len(paths):
            raise ValueError('Can not match "hashes" and "paths" lists, len(hashes) != len(paths)')

        h_to_path = {h: path for h, path in zip(hashes, paths)}
        for h, resp_part in self._download_batch_by_hashes(list(set(hashes))):
            ensure_base_path(h_to_path[h])
            with open(h_to_path[h], "wb") as w:
                w.write(resp_part.content)
            if progress_cb is not None:
                progress_cb(1)

    def download_nps_by_hashes_generator(
        self,
        hashes: List[str],
        keep_alpha: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        if len(hashes) == 0:
            return []

        if len(hashes) != len(set(hashes)):
            logger.warn("Found nonunique hashes in download task")

        for im_hash, resp_part in self._download_batch_by_hashes(hashes):
            yield im_hash, sly_image.read_bytes(resp_part.content, keep_alpha)
            if progress_cb is not None:
                progress_cb(1)

    def download_nps_by_hashes(
        self,
        hashes: List[str],
        keep_alpha: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[np.ndarray]:
        """
        Download Images with given hashes in Supervisely server in numpy format.

        :param hashes: List of images hashes in Supervisely.
        :type hashes: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: List of images
        :rtype: :class: List[np.ndarray]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_ids = [770918, 770919, 770920]
            image_hashes = []

            for img_id in image_ids:
                img_info = api.image.get_info_by_id(image_id)
                image_hashes.append(img_info.hash)

            image_nps = api.image.download_nps_by_hashes(image_hashes)
        """
        return [
            img
            for _, img in self.download_nps_by_hashes_generator(
                hashes,
                keep_alpha,
                progress_cb,
            )
        ]

    def get_project_id(self, image_id: int) -> int:
        """
        Gets Project ID by Image ID.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :return: Project ID where Image is located.
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_id = 121236920
            img_project_id = api.image.get_project_id(img_id)
            print(img_project_id)
            # Output: 53939
        """
        dataset_id = self.get_info_by_id(image_id, force_metadata_for_links=False).dataset_id
        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        return project_id

    @staticmethod
    def _get_free_name(exist_check_fn, name):
        """ """
        res_title = name
        suffix = 1

        name_without_ext = get_file_name(name)
        ext = get_file_ext(name)

        while exist_check_fn(res_title):
            res_title = "{}_{:03d}{}".format(name_without_ext, suffix, ext)
            suffix += 1
        return res_title

    def storage_url(self, path_original: str) -> str:
        """
        Get full Image URL link in Supervisely server.

        :param path_original: Original Image path in Supervisely server.
        :type path_original: str
        :return: Full Image URL link in Supervisely server
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 376729
            img_info = api.image.get_info_by_id(image_id)
            img_storage_url = api.image.storage_url(img_info.path_original)
        """

        return path_original

    def preview_url(
        self,
        url: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        quality: Optional[int] = 70,
    ) -> str:
        """
        Previews Image with the given resolution parameters.

        :param url: Full Image storage URL.
        :type url: str
        :param width: Preview Image width.
        :type width: int
        :param height: Preview Image height.
        :type height: int
        :param quality: Preview Image quality.
        :type quality: int
        :return: New URL with resized Image
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 376729
            img_info = api.image.get_info_by_id(image_id)
            img_preview_url = api.image.preview_url(img_info.full_storage_url, width=512, height=256)
        """
        # @TODO: if both width and height are defined, and they are not proportioned to original image resolution,
        # then images will be croped from center
        if width is None:
            width = ""
        if height is None:
            height = ""
        return url.replace(
            "/image-converter",
            f"/previews/{width}x{height},jpeg,q{quality}/image-converter",
        )

    def update_meta(self, id: int, meta: Dict) -> Dict:
        """
        It is possible to add custom JSON data to every image for storing some additional information.
        Updates Image metadata by ID. Metadata is visible in Labeling Tool.
        Supervisely also have 2 apps: import metadata and export metadata

        :param id: Image ID in Supervisely.
        :type id: int
        :param meta: Image metadata.
        :type meta: dict
        :raises: :class:`TypeError` if meta type is not dict
        :return: Image information in dict format with new meta
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import os
            import json
            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_info = api.image.get_info_by_id(id=3212008)
            print(image_info.meta)
            # Output: {}

            new_meta = {'Camera Make': 'Canon', 'Color Space': 'sRGB', 'Focal Length': '16 mm'}
            new_image_info = api.image.update_meta(id=3212008, meta=new_meta)

            image_info = api.image.get_info_by_id(id=3212008)
            print(json.dumps(obj=image_info.meta, indent=4))
            # Output: {
            #     "Camera Make": "Canon",
            #     "Color Space": "sRGB",
            #     "Focal Length": "16 mm"
            # }
        """
        if type(meta) is not dict:
            raise TypeError("Meta must be dict, not {}".format(type(meta)))
        response = self._api.post("images.editInfo", {ApiField.ID: id, ApiField.META: meta})
        return response.json()

    def add_tag(self, image_id: int, tag_id: int, value: Optional[Union[str, int]] = None) -> None:
        """
        Add tag with given ID to Image by ID.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :param tag_id: Tag ID in Supervisely.
        :type tag_id: int
        :param value: Tag value.
        :type value: int or str or None, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 2389126
            tag_id = 277083
            api.image.add_tag(image_id, tag_id)
        """

        self.add_tag_batch([image_id], tag_id, value)

    def add_tag_batch(
        self,
        image_ids: List[int],
        tag_id: int,
        value: Optional[Union[str, int]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        batch_size: Optional[int] = 100,
        tag_meta: Optional[TagMeta] = None,
    ) -> None:
        """
        Add tag with given ID to Images by IDs.

        :param image_ids: List of Images IDs in Supervisely.
        :type image_ids: List[int]
        :param tag_id: Tag ID in Supervisely.
        :type tag_id: int
        :param value: Tag value.
        :type value: int or str or None, optional
        :param progress_cb: Function for tracking progress of adding tag.
        :type progress_cb: tqdm or callable, optional
        :param batch_size: Batch size
        :type batch_size: int, optional
        :param tag_meta: Tag Meta. Needed for value validation, omit to skip validation
        :type tag_meta: TagMeta, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_ids = [2389126, 2389127]
            tag_id = 277083
            api.image.add_tag_batch(image_ids, tag_id)
        """
        if tag_meta:
            if not (tag_meta.sly_id == tag_id):
                raise ValueError("tag_meta.sly_id and tag_id should be same")
            if not tag_meta.is_valid_value(value):
                raise ValueError("Tag {} can not have value {}".format(tag_meta.name, value))
        else:
            # No value validation
            pass

        for batch_ids in batched(image_ids, batch_size):
            data = {ApiField.TAG_ID: tag_id, ApiField.IDS: batch_ids}
            if value is not None:
                data[ApiField.VALUE] = value
            self._api.post("image-tags.bulk.add-to-image", data)
            if progress_cb is not None:
                progress_cb(len(batch_ids))

    def remove_batch(
        self,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        batch_size: Optional[int] = 50,
    ):
        """
        Remove images from supervisely by ids.

        :param ids: List of Images IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Function for tracking progress of removing.
        :type progress_cb: tqdm or callable, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_ids = [2389126, 2389127]
            api.image.remove_batch(image_ids)
        """
        super(ImageApi, self).remove_batch(ids, progress_cb=progress_cb, batch_size=batch_size)

    def remove(self, image_id: int):
        """
        Remove image from supervisely by id.

        :param image_id: Images ID in Supervisely.
        :type image_id: int
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 2389126
            api.image.remove(image_id)
        """
        super(ImageApi, self).remove(image_id)

    def exists(self, parent_id, name):
        """exists"""
        return self.get_info_by_name(parent_id, name, force_metadata_for_links=False) is not None
