# coding: utf-8
from __future__ import annotations
import json
from typing import Callable, Dict, List, Optional, Union

from tqdm import tqdm

from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.api.module_api import ApiField
from supervisely.io.json import load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation


class VideoAnnotationAPI(EntityAnnotationAPI):
    """
    :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` for a single video. :class:`VideoAnnotationAPI<VideoAnnotationAPI>` object is immutable.

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

        video_id = 186648102
        ann_info = api.video.annotation.download(video_id)
    """

    _method_download_bulk = "videos.annotations.bulk.info"
    _entity_ids_str = ApiField.VIDEO_IDS

    def download(self, video_id: int) -> Dict:
        """
        Download information about VideoAnnotation by video ID from API.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :return: Information about VideoAnnotation in json format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198702499
            ann_info = api.video.annotation.download(video_id)
            print(ann_info)
            # Output: {
            #     "videoId": 198702499,
            #     "videoName": "Videos_dataset_cars_cars.mp4",
            #     "createdAt": "2021-03-23T13:14:25.536Z",
            #     "updatedAt": "2021-03-23T13:16:43.300Z",
            #     "description": "",
            #     "tags": [],
            #     "objects": [],
            #     "size": {
            #         "height": 2160,
            #         "width": 3840
            #     },
            #     "framesCount": 326,
            #     "frames": []
            # }
        """

        video_info = self._api.video.get_info_by_id(video_id)
        return self._download(video_info.dataset_id, video_id)

    def append(
        self,
        video_id: int,
        ann: VideoAnnotation,
        key_id_map: Optional[KeyIdMap] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Loads an VideoAnnotation to a given video ID in the API.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param ann: VideoAnnotation object.
        :type ann: VideoAnnotation
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :param progress: Progress.
        :type progress: Optional[Union[tqdm, Callable]]
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198704259
            api.video.annotation.append(video_id, video_ann)
        """

        info = self._api.video.get_info_by_id(video_id)
        self._append(
            self._api.video.tag,
            self._api.video.object,
            self._api.video.figure,
            info.project_id,
            info.dataset_id,
            video_id,
            ann.tags,
            ann.objects,
            ann.figures,
            key_id_map,
            progress_cb,
        )

    def upload_paths(
        self,
        video_ids: List[int],
        ann_paths: List[str],
        project_meta: ProjectMeta,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Loads an VideoAnnotations from a given paths to a given videos IDs in the API. Videos IDs must be from one dataset.

        :param video_ids: Videos IDs in Supervisely.
        :type video_ids: List[int]
        :param ann_paths: Paths to annotations on local machine.
        :type ann_paths: List[str]
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>` for VideoAnnotations.
        :type project_meta: ProjectMeta
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

            video_ids = [121236918, 121236919]
            ann_paths = ['/home/admin/work/supervisely/example/ann1.json', '/home/admin/work/supervisely/example/ann2.json']
            api.video.annotation.upload_paths(video_ids, ann_paths, meta)
        """
        # video_ids from the same dataset

        for video_id, ann_path in zip(video_ids, ann_paths):
            ann_json = load_json_file(ann_path)
            ann = VideoAnnotation.from_json(ann_json, project_meta)

            # ignore existing key_id_map because the new objects will be created
            self.append(video_id, ann)
