# coding: utf-8
"""download/upload/edit :class:`Annotation<supervisely.annotation.annotation.Annotation>`"""

# docs
from __future__ import annotations
import json
from typing import List, Optional, NamedTuple, Dict, Callable, Union
from supervisely.task.progress import Progress
from supervisely.annotation.label import Label

from supervisely.annotation.annotation import Annotation
from supervisely.api.module_api import ApiField, ModuleApi
from supervisely._utils import batched
from tqdm import tqdm


class AnnotationInfo(NamedTuple):
    """
    AnnotationInfo
    """

    image_id: int
    image_name: str
    annotation: dict
    created_at: str
    updated_at: str


class AnnotationApi(ModuleApi):
    """
    Annotation for a single image. :class:`AnnotationApi<AnnotationApi>` object is immutable.

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

        dataset_id = 254737
        ann_infos = api.annotation.get_list(dataset_id)
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple AnnotationInfo information about Annotation.

        :Example:

         .. code-block:: python

            AnnotationInfo(image_id=121236919,
                           image_name='IMG_1836',
                           annotation={'description': '', 'tags': [], 'size': {'height': 800, 'width': 1067}, 'objects': []},
                           created_at='2019-12-19T12:06:59.435Z',
                           updated_at='2021-02-06T11:07:26.080Z')
        """
        return [
            ApiField.IMAGE_ID,
            ApiField.IMAGE_NAME,
            ApiField.ANNOTATION,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **AnnotationInfo**.
        """
        return "AnnotationInfo"

    def get_list(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links: Optional[bool] = True,
    ) -> List[AnnotationInfo]:
        """
        Get list of information about all annotations for a given dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Annotations.
        :type filters: List[dict], optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[AnnotationInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            ann_infos = api.annotation.get_list(dataset_id)
            print(json.dumps(ann_infos[0], indent=4))
            # Output: [
            #     121236918,
            #     "IMG_0748.jpeg",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]

            ann_infos_filter = api.annotation.get_list(dataset_id, filters={ 'field': 'name', 'operator': '=', 'value': 'IMG_1836' })
            print(json.dumps(ann_infos_filter, indent=4))
            # Output: [
            #     121236919,
            #     "IMG_1836",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]
        """
        return self.get_list_all_pages(
            "annotations.list",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.FILTER: filters or [],
                ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            },
            progress_cb,
        )

    def get_list_generator(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        batch_size: Optional[int] = 50,
        force_metadata_for_links: Optional[bool] = True,
    ) -> List[AnnotationInfo]:
        """
        Get list of information about all annotations for a given dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Annotations.
        :type filters: List[dict], optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[AnnotationInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            ann_infos = api.annotation.get_list(dataset_id)
            print(json.dumps(ann_infos[0], indent=4))
            # Output: [
            #     121236918,
            #     "IMG_0748.jpeg",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]

            ann_infos_filter = api.annotation.get_list(dataset_id, filters={ 'field': 'name', 'operator': '=', 'value': 'IMG_1836' })
            print(json.dumps(ann_infos_filter, indent=4))
            # Output: [
            #     121236919,
            #     "IMG_1836",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]
        """
        data = {
            ApiField.DATASET_ID: dataset_id,
            ApiField.FILTER: filters or [],
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            ApiField.PAGINATION_MODE: ApiField.TOKEN,
        }
        if batch_size is not None:
            data[ApiField.PER_PAGE] = batch_size
        else:
            # use default value on instance (learn in API documentation)
            # 20k for instance
            # 50 by default in SDK
            pass

        return self.get_list_all_pages_generator("annotations.list", data, progress_cb)

    def download(
        self,
        image_id: int,
        with_custom_data: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
    ) -> AnnotationInfo:
        """
        Download AnnotationInfo by image ID from API.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :param with_custom_data:
        :type with_custom_data: bool, optional
        :return: Information about Annotation. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`AnnotationInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            ann_info = api.annotation.download(image_id)
            print(json.dumps(ann_info, indent=4))
            # Output: [
            #     121236918,
            #     "IMG_0748.jpeg",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]
        """
        response = self._api.post(
            "annotations.info",
            {
                ApiField.IMAGE_ID: image_id,
                ApiField.WITH_CUSTOM_DATA: with_custom_data,
                ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            },
        )
        return self._convert_json_info(response.json())

    def download_json(
        self,
        image_id: int,
        with_custom_data: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
    ) -> Dict[str, Union[str, int, list, dict]]:
        """
        Download Annotation in json format by image ID from API.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :param with_custom_data:
        :type with_custom_data: bool, optional
        :return: Annotation in json format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            ann_json = api.annotation.download_json(image_id)
            print(ann_json)
            # Output: {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     }
        """
        return self.download(
            image_id=image_id,
            with_custom_data=with_custom_data,
            force_metadata_for_links=force_metadata_for_links,
        ).annotation

    def download_batch(
        self,
        dataset_id: int,
        image_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        with_custom_data: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
    ) -> List[AnnotationInfo]:
        """
        Get list of AnnotationInfos for given dataset ID from API.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: List of integers.
        :type image_ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm
        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[AnnotationInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            image_ids = [121236918, 121236919]
            p = tqdm(desc="Annotations downloaded: ", total=len(image_ids))

            ann_infos = api.annotation.download_batch(dataset_id, image_ids, progress_cb=p)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations downloaded: ", "current": 0, "total": 2, "timestamp": "2021-03-16T15:20:06.168Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations downloaded: ", "current": 2, "total": 2, "timestamp": "2021-03-16T15:20:06.510Z", "level": "info"}
        """
        id_to_ann = {}
        for batch in batched(image_ids):
            post_data = {
                ApiField.DATASET_ID: dataset_id,
                ApiField.IMAGE_IDS: batch,
                ApiField.WITH_CUSTOM_DATA: with_custom_data,
                ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            }
            results = self._api.post("annotations.bulk.info", data=post_data).json()
            for ann_dict in results:
                ann_info = self._convert_json_info(ann_dict)
                id_to_ann[ann_info.image_id] = ann_info
            if progress_cb is not None:
                progress_cb(len(batch))
        ordered_results = [id_to_ann[image_id] for image_id in image_ids]
        return ordered_results

    def download_json_batch(
        self,
        dataset_id: int,
        image_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links: Optional[bool] = True,
    ) -> List[Dict]:
        """
        Get list of AnnotationInfos for given dataset ID from API.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: List of integers.
        :type image_ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm
        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[Dict]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            image_ids = [121236918, 121236919]
            p = tqdm(desc="Annotations downloaded: ", total=len(image_ids))

            anns_jsons = api.annotation.download_json_batch(dataset_id, image_ids, progress_cb=p)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations downloaded: ", "current": 0, "total": 2, "timestamp": "2021-03-16T15:20:06.168Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations downloaded: ", "current": 2, "total": 2, "timestamp": "2021-03-16T15:20:06.510Z", "level": "info"}
        """
        results = self.download_batch(
            dataset_id=dataset_id,
            image_ids=image_ids,
            progress_cb=progress_cb,
            force_metadata_for_links=force_metadata_for_links,
        )
        return [ann_info.annotation for ann_info in results]

    def upload_path(
        self,
        img_id: int,
        ann_path: str,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an annotation from a given path to a given image ID in the API.

        :param img_id: Image ID in Supervisely.
        :type img_id: int
        :param ann_path: Path to annotation on host.
        :type ann_path: str
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            ann_path = '/home/admin/work/supervisely/example/ann.json'
            upl_path = api.annotation.upload_path(image_id, ann_path)
        """
        self.upload_paths([img_id], [ann_path], skip_bounds_validation=skip_bounds_validation)

    def upload_paths(
        self,
        img_ids: List[int],
        ann_paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an annotations from a given paths to a given images IDs in the API. Images IDs must be from one dataset.

        :param img_ids: Images IDs in Supervisely.
        :type img_ids: List[int]
        :param ann_paths: Paths to annotations on local machine.
        :type ann_paths: List[str]
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

            img_ids = [121236918, 121236919]
            ann_pathes = ['/home/admin/work/supervisely/example/ann1.json', '/home/admin/work/supervisely/example/ann2.json']
            upl_paths = api.annotation.upload_paths(img_ids, ann_pathes)
        """

        def read_json(ann_path):
            with open(ann_path) as json_file:
                return json.load(json_file)

        self._upload_batch(
            read_json,
            img_ids,
            ann_paths,
            progress_cb,
            skip_bounds_validation=skip_bounds_validation,
        )

    def upload_json(
        self,
        img_id: int,
        ann_json: Dict,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an annotation from dict to a given image ID in the API.

        :param img_id: Image ID in Supervisely.
        :type img_id: int
        :param ann_json: Annotation in JSON format.
        :type ann_json: dict
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            upl_json = api.annotation.upload_json(image_id, ann_json)
        """
        self.upload_jsons([img_id], [ann_json], skip_bounds_validation=skip_bounds_validation)

    def upload_jsons(
        self,
        img_ids: List[int],
        ann_jsons: List[Dict],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an annotations from dicts to a given images IDs in the API. Images IDs must be from one dataset.

        :param img_ids: Image ID in Supervisely.
        :type img_ids: List[int]
        :param ann_jsons: Annotation in JSON format.
        :type ann_jsons: List[dict]
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

            img_ids = [121236918, 121236919]
            upl_jsons = api.annotation.upload_jsons(img_ids, ann_jsons)
        """
        self._upload_batch(
            lambda x: x,
            img_ids,
            ann_jsons,
            progress_cb,
            skip_bounds_validation=skip_bounds_validation,
        )

    def upload_ann(
        self,
        img_id: int,
        ann: Annotation,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an :class:`Annotation<supervisely.annotation.annotation.Annotation>` to a given image ID in the API.

        :param img_id: Image ID in Supervisely.
        :type img_id: int
        :param ann: Annotation object.
        :type ann: Annotation
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            upl_ann = api.annotation.upload_ann(image_id, ann)
        """
        self.upload_anns([img_id], [ann], skip_bounds_validation=skip_bounds_validation)

    def upload_anns(
        self,
        img_ids: List[int],
        anns: List[Annotation],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an :class:`Annotations<supervisely.annotation.annotation.Annotation>` to a given images IDs in the API. Images IDs must be from one dataset.

        :param img_ids: Image ID in Supervisely.
        :type img_ids: List[int]
        :param anns: List of Annotation objects.
        :type anns: List[Annotation]
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

            img_ids = [121236918, 121236919]
            upl_anns = api.annotation.upload_anns(img_ids, [ann1, ann2])
        """
        # img_ids from the same dataset
        self._upload_batch(
            Annotation.to_json,
            img_ids,
            anns,
            progress_cb,
            skip_bounds_validation=skip_bounds_validation,
        )

    def _upload_batch(
        self,
        func_ann_to_json,
        img_ids,
        anns,
        progress_cb=None,
        skip_bounds_validation: Optional[bool] = False,
    ):
        """
        _upload_batch
        """
        # img_ids from the same dataset
        if len(img_ids) == 0:
            return
        if len(img_ids) != len(anns):
            raise RuntimeError(
                'Can not match "img_ids" and "anns" lists, len(img_ids) != len(anns)'
            )

        dataset_id = self._api.image.get_info_by_id(
            img_ids[0], force_metadata_for_links=False
        ).dataset_id
        for batch in batched(list(zip(img_ids, anns))):
            data = [
                {ApiField.IMAGE_ID: img_id, ApiField.ANNOTATION: func_ann_to_json(ann)}
                for img_id, ann in batch
            ]
            self._api.post(
                "annotations.bulk.add",
                data={
                    ApiField.DATASET_ID: dataset_id,
                    ApiField.ANNOTATIONS: data,
                    ApiField.SKIP_BOUNDS_VALIDATION: skip_bounds_validation,
                },
            )
            if progress_cb is not None:
                progress_cb(len(batch))

    def get_info_by_id(self, id):
        """
        get_info_by_id
        """
        raise NotImplementedError("Method is not supported")

    def get_info_by_name(self, parent_id, name):
        """
        get_info_by_name
        """
        raise NotImplementedError("Method is not supported")

    def exists(self, parent_id, name):
        """
        exists
        """
        raise NotImplementedError("Method is not supported")

    def get_free_name(self, parent_id, name):
        """
        get_free_name
        """
        raise NotImplementedError("Method is not supported")

    def _add_sort_param(self, data):
        """
        _add_sort_param
        """
        return data

    def copy_batch(
        self,
        src_image_ids: List[int],
        dst_image_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links: Optional[bool] = True,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Copy annotations from one images IDs to another in API.

        :param src_image_ids: Images IDs in Supervisely.
        :type src_image_ids: List[int]
        :param dst_image_ids: Unique IDs of images in API.
        :type dst_image_ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`RuntimeError`, if len(src_image_ids) != len(dst_image_ids)
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from tqdm import tqdm

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_ids = [121236918, 121236919]
            dst_ids = [547837053, 547837054]
            p = tqdm(desc="Annotations copy: ", total=len(src_ids))

            copy_anns = api.annotation.copy_batch(src_ids, dst_ids, progress_cb=p)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations copy: ", "current": 0, "total": 2, "timestamp": "2021-03-16T15:24:31.286Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations copy: ", "current": 2, "total": 2, "timestamp": "2021-03-16T15:24:31.288Z", "level": "info"}
        """
        if len(src_image_ids) != len(dst_image_ids):
            raise RuntimeError(
                'Can not match "src_image_ids" and "dst_image_ids" lists, '
                "len(src_image_ids) != len(dst_image_ids)"
            )
        if len(src_image_ids) == 0:
            return

        src_dataset_id = self._api.image.get_info_by_id(src_image_ids[0]).dataset_id
        for cur_batch in batched(list(zip(src_image_ids, dst_image_ids))):
            src_ids_batch, dst_ids_batch = zip(*cur_batch)
            ann_infos = self.download_batch(
                src_dataset_id, src_ids_batch, force_metadata_for_links=force_metadata_for_links
            )
            ann_jsons = [ann_info.annotation for ann_info in ann_infos]
            self.upload_jsons(
                dst_ids_batch, ann_jsons, skip_bounds_validation=skip_bounds_validation
            )
            if progress_cb is not None:
                progress_cb(len(src_ids_batch))

    def copy(
        self,
        src_image_id: int,
        dst_image_id: int,
        force_metadata_for_links: Optional[bool] = True,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Copy annotation from one image ID to another image ID in API.

        :param src_image_id: Image ID in Supervisely.
        :type src_image_id: int
        :param dst_image_id: Image ID in Supervisely.
        :type dst_image_id: int
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_id = 121236918
            dst_id = 547837053
            api.annotation.copy(src_id, dst_id)
        """
        self.copy_batch(
            [src_image_id],
            [dst_image_id],
            force_metadata_for_links=force_metadata_for_links,
            skip_bounds_validation=skip_bounds_validation,
        )

    def copy_batch_by_ids(
        self,
        src_image_ids: List[int],
        dst_image_ids: List[int],
        batch_size: Optional[int] = 50,
        save_source_date: Optional[bool] = True,
    ) -> None:
        """
        Copy annotations from one images IDs to another images IDs in API.

        :param src_image_ids: Images IDs in Supervisely.
        :type src_image_ids: List[int]
        :param dst_image_ids: Images IDs in Supervisely.
        :type dst_image_ids: List[int]
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`RuntimeError` if len(src_image_ids) != len(dst_image_ids)

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_ids = [121236918, 121236919]
            dst_ids = [547837053, 547837054]
            api.annotation.copy_batch_by_ids(src_ids, dst_ids)
        """
        if len(src_image_ids) != len(dst_image_ids):
            raise RuntimeError(
                'Can not match "src_image_ids" and "dst_image_ids" lists, '
                "len(src_image_ids) != len(dst_image_ids)"
            )
        if len(src_image_ids) == 0:
            return
        for cur_batch in batched(list(zip(src_image_ids, dst_image_ids)), batch_size=batch_size):
            src_ids_batch, dst_ids_batch = zip(*cur_batch)
            self._api.post(
                "annotations.bulk.copy",
                data={
                    "srcImageIds": src_ids_batch,
                    "destImageIds": dst_ids_batch,
                    "preserveSourceDate": save_source_date,
                },
            )

    def _convert_json_info(self, info: dict, skip_missing=True) -> AnnotationInfo:
        """
        _convert_json_info
        """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return AnnotationInfo(**res._asdict())

    def append_labels(
        self,
        image_id: int,
        labels: List[Label],
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Append labels to image with given ID in API.

        :param image_id: Image ID to append labels.
        :type image_id: int
        :param labels: List of labels to append.
        :type labels: List[Label]
        :return: None
        :rtype: :class:`NoneType`
        """
        if len(labels) == 0:
            return

        payload = []
        for label in labels:
            _label_json = label.to_json()
            _label_json["geometry"] = label.geometry.to_json()
            if "classId" not in _label_json:
                raise KeyError("Update project meta from server to get class id")
            payload.append(_label_json)

        added_ids = []
        for batch_jsons in batched(payload, batch_size=100):
            resp = self._api.post(
                "figures.bulk.add",
                {
                    ApiField.ENTITY_ID: image_id,
                    ApiField.FIGURES: batch_jsons,
                    ApiField.SKIP_BOUNDS_VALIDATION: skip_bounds_validation,
                },
            )
            for resp_obj in resp.json():
                figure_id = resp_obj[ApiField.ID]
                added_ids.append(figure_id)
