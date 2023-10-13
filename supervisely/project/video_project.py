# coding: utf-8

# docs
from __future__ import annotations
from collections import namedtuple
import os
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

from tqdm import tqdm
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.video.video_api import VideoInfo
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.io.fs import file_exists, touch, mkdir
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project import (
    Dataset,
    OpenMode,
    Project,
    read_single_project as read_project_wrapper,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress
from supervisely._utils import batched
from supervisely.video import video as sly_video
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation


class VideoItemPaths(NamedTuple):
    #: :class:`str`: Full video file path of item
    video_path: str

    #: :class:`str`: Full annotation file path of item
    ann_path: str


class VideoDataset(Dataset):
    """
    VideoDataset is where your labeled and unlabeled videos and other data files live. :class:`VideoDataset<VideoDataset>` object is immutable.

    :param directory: Path to dataset directory.
    :type directory: str
    :param mode: Determines working mode for the given dataset.
    :type mode: :class:`OpenMode<supervisely.project.project.OpenMode>`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
        ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)
    """

    #: :class:`str`: Items data directory name
    item_dir_name = "video"

    #: :class:`str`: Annotations directory name
    ann_dir_name = "ann"

    #: :class:`str`: Items info directory name
    item_info_dir_name = "video_info"

    #: :class:`str`: Segmentation masks directory name
    seg_dir_name = None

    annotation_class = VideoAnnotation
    item_info_class = VideoInfo

    @property
    def project_dir(self) -> str:
        """
        Path to the video project containing the video dataset.

        :return: Path to the video project.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)
            print(ds.project_dir)
            # Output: "/home/admin/work/supervisely/projects/videos_example"
        """
        return super().project_dir

    @property
    def name(self) -> str:
        """
        Video Dataset name.

        :return: Video Dataset Name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)
            print(ds.name)
            # Output: "ds0"
        """
        return super().name

    @property
    def directory(self) -> str:
        """
        Path to the video dataset directory.

        :return: Path to the video dataset directory.
        :rtype: :class:`str`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.directory)
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0'
        """
        return super().directory

    @property
    def item_dir(self) -> str:
        """
        Path to the video dataset items directory.

        :return: Path to the video dataset items directory.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.item_dir)
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/video'
        """
        return super().item_dir

    @property
    def img_dir(self) -> str:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Property 'img_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def ann_dir(self) -> str:
        """
        Path to the video dataset annotations directory.

        :return: Path to the video dataset directory with annotations.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.ann_dir)
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/ann'
        """
        return super().ann_dir

    @property
    def img_info_dir(self):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Property 'img_info_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def item_info_dir(self):
        """
        Path to the video dataset item with items info.

        :return: Path to the video dataset directory with items info.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.item_info_dir)
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/video_info'
        """
        return super().item_info_dir

    @property
    def seg_dir(self):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Property 'seg_dir' is not supported for {type(self).__name__} object."
        )

    @classmethod
    def _has_valid_ext(cls, path: str) -> bool:
        """
        Checks if file from given path is supported
        :param path: str
        :return: bool
        """
        return sly_video.has_valid_ext(path)

    def get_items_names(self) -> list:
        """
        List of video dataset item names.

        :return: List of item names.
        :rtype: :class:`list` [ :class:`str` ]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_names())
            # Output: ['video_0002.mp4', 'video_0005.mp4', 'video_0008.mp4', ...]
        """
        return super().get_items_names()

    def item_exists(self, item_name: str) -> bool:
        """
        Checks if given item name belongs to the video dataset.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: True if item exist, otherwise False.
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            ds.item_exists("video_0748")     # False
            ds.item_exists("video_0748.mp4") # True
        """
        return super().item_exists(item_name)

    def get_item_path(self, item_name: str) -> str:
        """
        Path to the given item.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: Path to the given item.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_path("video_0748"))
            # Output: RuntimeError: Item video_0748 not found in the project.

            print(ds.get_item_path("video_0748.mp4"))
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/video/video_0748.mp4'
        """
        return super().get_item_path(item_name)

    def get_img_path(self, item_name: str) -> str:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_img_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_ann(
        self, item_name, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> VideoAnnotation:
        """
        Read annotation of item from json.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param project_meta: ProjectMeta object.
        :type project_meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: VideoAnnotation object.
        :rtype: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/videos_example"
            project = sly.VideoProject(project_path, sly.OpenMode.READ)

            ds = project.datasets.get('ds0')

            annotation = ds.get_ann("video_0748", project.meta)
            # Output: RuntimeError: Item video_0748 not found in the project.

            annotation = ds.get_ann("video_0748.mp4", project.meta)
            print(annotation.to_json())
            # Output: {
            #     "description": "",
            #     "size": {
            #         "height": 500,
            #         "width": 700
            #     },
            #     "key": "e9ef52dbbbbb490aa10f00a50e1fade6",
            #     "tags": [],
            #     "objects": [],
            #     "frames": [{
            #         "index": 0,
            #         "figures": []
            #     }]
            #     "framesCount": 1
            # }
        """
        ann_path = self.get_ann_path(item_name)
        return self.annotation_class.load_json_file(ann_path, project_meta, key_id_map)

    def get_ann_path(self, item_name: str) -> str:
        """
        Path to the given annotation json file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: Path to the given annotation json file.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_ann_path("video_0748"))
            # Output: RuntimeError: Item video_0748 not found in the project.

            print(ds.get_ann_path("video_0748.mp4"))
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/ann/video_0748.mp4.json'
        """
        return super().get_ann_path(item_name)

    def get_img_info_path(self, img_name: str) -> str:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_img_info_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_item_info_path(self, item_name: str) -> str:
        """
        Get path to the item info json file without checking if the file exists.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: Path to the given item info json file.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_info_path("video_0748"))
            # Output: RuntimeError: Item video_0748 not found in the project.

            print(ds.get_item_info_path("video_0748.mp4"))
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/video_info/video_0748.mp4.json'
        """
        return super().get_item_info_path(item_name)

    def get_image_info(self, item_name: str) -> None:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_image_info(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_item_info(self, item_name: str) -> VideoInfo:
        """
        Information for Item with given name.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: VideoInfo object.
        :rtype: :class:`VideoInfo<supervisely.api.video.video_api.VideoInfo>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_info("video_0748.mp4"))
            # Output:
            # VideoInfo(
            #     id=198702499,
            #     name='video_0748.mp4',
            #     hash='ehYHLNFWmMNuF2fPUgnC/g/tkIIEjNIOhdbNLQXkE8Y=',
            #     team_id=16087,
            #     workspace_id=23821,
            #     project_id=124974,
            #     dataset_id=466639,
            #     path_original='/h5un6l2bnaz1vj8a9qgms4-public/videos/w/7/i4/GZYoCs...9F3kyVJ7.mp4',
            #     frames_to_timecodes=[0, 0.033367, 0.066733, 0.1001,...,10.777433, 10.8108, 10.844167],
            #     frames_count=326,
            #     frame_width=3840,
            #     frame_height=2160,
            #     created_at='2021-03-23T13:14:25.536Z',
            #     updated_at='2021-03-23T13:16:43.300Z'
            # )
        """
        item_info_path = self.get_item_info_path(item_name)
        item_info_dict = load_json_file(item_info_path)
        item_info_named_tuple = namedtuple(self.item_info_class.__name__, item_info_dict)
        return item_info_named_tuple(**item_info_dict)

    def get_seg_path(self, item_name: str) -> str:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_seg_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def add_item_file(
        self,
        item_name: str,
        item_path: str,
        ann: Optional[Union[VideoAnnotation, str]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        item_info: Optional[Union[VideoInfo, Dict, str]] = None,
    ) -> None:
        """
        Adds given item file to dataset items directory, and adds given annotation to dataset
        annotations directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param item_path: Path to the item.
        :type item_path: :class:`str`
        :param ann: VideoAnnotation object or path to annotation json file.
        :type ann: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` or :class:`str`, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: :class:`bool`, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: :class:`bool`, optional
        :param item_info: VideoInfo object or VideoInfo object converted to dict or path to item info json file for copying to dataset item info directory.
        :type item_info: :class:`VideoInfo<supervisely.api.video.video_api.VideoInfo>` or :class:`dict` or :class:`str`, optional
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if item_name already exists in dataset or item name has unsupported extension.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            ann = "/home/admin/work/supervisely/projects/videos_example/ds0/ann/video_8888.mp4.json"
            ds.add_item_file("video_8888.mp4", "/home/admin/work/supervisely/projects/videos_example/ds0/video/video_8888.mp4", ann=ann)
            print(ds.item_exists("video_8888.mp4"))
            # Output: True
        """
        return super().add_item_file(
            item_name=item_name,
            item_path=item_path,
            ann=ann,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
            item_info=item_info,
        )

    def add_item_np(self, item_name, img, ann=None, img_info=None):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'add_item_np()' is not supported for {type(self).__name__} object."
        )

    def add_item_raw_bytes(self, item_name, item_raw_bytes, ann=None, img_info=None):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def get_classes_stats(
        self,
        project_meta: Optional[ProjectMeta] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        if project_meta is None:
            project = VideoProject(self.project_dir, OpenMode.READ)
            project_meta = project.meta
        class_items = {}
        class_objects = {}
        class_figures = {}
        for obj_class in project_meta.obj_classes:
            class_items[obj_class.name] = 0
            class_objects[obj_class.name] = 0
            class_figures[obj_class.name] = 0
        for item_name in self:
            item_ann = self.get_ann(item_name, project_meta)
            item_class = {}
            for ann_obj in item_ann.objects:
                class_objects[ann_obj.obj_class.name] += 1
            for video_figure in item_ann.figures:
                class_figures[video_figure.parent_object.obj_class.name] += 1
                item_class[video_figure.parent_object.obj_class.name] = True
            for obj_class in project_meta.obj_classes:
                if obj_class.name in item_class.keys():
                    class_items[obj_class.name] += 1

        result = {}
        if return_items_count:
            result["items_count"] = class_items
        if return_objects_count:
            result["objects_count"] = class_objects
        if return_figures_count:
            result["figures_count"] = class_figures
        return result

    def _get_empty_annotaion(self, item_name):
        """
        Create empty VideoAnnotation for given video
        :param item_name: str
        :return: VideoAnnotation class object
        """
        img_size, frames_count = sly_video.get_image_size_and_frames_count(item_name)
        return self.annotation_class(img_size, frames_count)

    def _add_item_raw_bytes(self, item_name, item_raw_bytes):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method '_add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def _add_img_np(self, item_name, img):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method '_add_img_np()' is not supported for {type(self).__name__} object."
        )

    def _validate_added_item_or_die(self, item_path):
        """
        Make sure we actually received a valid video file, clean it up and fail if not so.
        :param item_path: str
        """
        # Make sure we actually received a valid video file, clean it up and fail if not so.
        try:
            sly_video.validate_format(item_path)
        except Exception as e:
            os.remove(item_path)
            raise e

    def set_ann(
        self, item_name: str, ann: VideoAnnotation, key_id_map: Optional[KeyIdMap] = None
    ) -> None:
        """
        Replaces given annotation for given item name to dataset annotations directory in json format.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param ann: VideoAnnotation object.
        :type ann: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            height, width = 500, 700
            new_ann = sly.VideoAnnotation((height, width), frames_count=0)
            ds.set_ann("video_0748.mp4", new_ann)
        """
        if type(ann) is not self.annotation_class:
            raise TypeError(
                f"Type of 'ann' should be {self.annotation_class.__name__}, not a {type(ann).__name__}"
            )
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann.to_json(key_id_map), dst_ann_path, indent=4)

    def get_item_paths(self, item_name) -> VideoItemPaths:
        """
        Generates :class:`VideoItemPaths<VideoItemPaths>` object with paths to item and annotation directories for item with given name.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: VideoItemPaths object
        :rtype: :class:`VideoItemPaths<VideoItemPaths>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            video_path, ann_path = dataset.get_item_paths("video_0748.mp4")
            print("video_path:", video_path)
            print("ann_path:", ann_path)
            # Output:
            # video_path: /home/admin/work/supervisely/projects/videos_example/ds0/video/video_0748.mp4
            # ann_path: /home/admin/work/supervisely/projects/videos_example/ds0/ann/video_0748.mp4.json
        """
        return VideoItemPaths(
            video_path=self.get_item_path(item_name), ann_path=self.get_ann_path(item_name)
        )

    @staticmethod
    def get_url(project_id: int, dataset_id: int) -> str:
        """
        Get URL to dataset items list in Supervisely.

        :param project_id: :class:`VideoProject<VideoProject>` ID in Supervisely.
        :type project_id: :class:`int`
        :param dataset_id: :class:`VideoDataset<VideoDataset>` ID in Supervisely.
        :type dataset_id: :class:`int`
        :return: URL to dataset items list.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely import VideoDataset

            project_id = 10093
            dataset_id = 45330
            ds_items_link = VideoDataset.get_url(project_id, dataset_id)

            print(ds_items_link)
            # Output: "/projects/10093/datasets/45330/entities"
        """
        return super().get_url(project_id, dataset_id)


class VideoProject(Project):
    """
    VideoProject is a parent directory for video dataset. VideoProject object is immutable.

    :param directory: Path to video project directory.
    :type directory: :class:`str`
    :param mode: Determines working mode for the given project.
    :type mode: :class:`OpenMode<supervisely.project.project.OpenMode>`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        project_path = "/home/admin/work/supervisely/projects/videos_example"
        project = sly.Project(project_path, sly.OpenMode.READ)
    """

    dataset_class = VideoDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = VideoDataset

    def __init__(self, directory, mode: OpenMode):
        """
        :param directory: path to the directory where the project will be saved or where it will be loaded from
        :param mode: OpenMode class object which determines in what mode to work with the project (generate exception error if not so)
        """
        self._key_id_map: KeyIdMap = None
        super().__init__(directory, mode)

    @staticmethod
    def get_url(id: int) -> str:
        """
        Get URL to video datasets list in Supervisely.

        :param id: :class:`VideoProject<VideoProject>` ID in Supervisely.
        :type id: :class:`int`
        :return: URL to datasets list.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely import VideoProject

            project_id = 10093
            datasets_link = VideoProject.get_url(project_id)

            print(datasets_link)
            # Output: "/projects/10093/datasets"
        """
        return super().get_url(id)

    def get_classes_stats(
        self,
        dataset_names: Optional[List[str]] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        return super(VideoProject, self).get_classes_stats(
            dataset_names, return_objects_count, return_figures_count, return_items_count
        )

    def _read(self):
        """
        Download project from given project directory. Checks item and annotation directoris existing and dataset not empty.
        Consistency checks. Every video must have an annotation, and the correspondence must be one to one.
        """
        super()._read()
        self._key_id_map = KeyIdMap()
        if os.path.exists(self._get_key_id_map_path()):
            self._key_id_map = self._key_id_map.load_json(self._get_key_id_map_path())

    def _create(self):
        """
        Creates a leaf directory and empty meta.json file. Generate exception error if project directory already exists and is not empty.
        """
        super()._create()
        self.set_key_id_map(KeyIdMap())

    @property
    def key_id_map(self):
        # TODO: write docstring
        return self._key_id_map

    def set_key_id_map(self, new_map: KeyIdMap):
        """
        Save given KeyIdMap object to project dir in json format.
        :param new_map: KeyIdMap class object
        """
        self._key_id_map = new_map
        self._key_id_map.dump_json(self._get_key_id_map_path())

    def _get_key_id_map_path(self):
        """
        :return: str (full path to key_id_map.json)
        """
        return os.path.join(self.directory, "key_id_map.json")

    def copy_data(
        self,
        dst_directory: str,
        dst_name: Optional[str] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
    ) -> VideoProject:
        """
        Makes a copy of the :class:`VideoProject<VideoProject>`.

        :param dst_directory: Path to video project parent directory.
        :type dst_directory: :class:`str`
        :param dst_name: Video Project name.
        :type dst_name: :class:`str`, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: :class:`bool`, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: :class:`bool`, optional
        :return: VideoProject object.
        :rtype: :class:`VideoProject<VideoProject>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.VideoProject("/home/admin/work/supervisely/projects/videos_example", sly.OpenMode.READ)
            print(project.total_items)
            # Output: 6

            new_project = project.copy_data("/home/admin/work/supervisely/projects/", "videos_example_copy")
            print(new_project.total_items)
            # Output: 6
        """
        dst_name = dst_name if dst_name is not None else self.name
        new_project = VideoProject(os.path.join(dst_directory, dst_name), OpenMode.CREATE)
        new_project.set_meta(self.meta)

        for ds in self:
            new_ds = new_project.create_dataset(ds.name)

            for item_name in ds:
                item_path, ann_path = ds.get_item_paths(item_name)
                item_info_path = ds.get_item_info_path(item_name)

                item_path = item_path if os.path.isfile(item_path) else None
                ann_path = ann_path if os.path.isfile(ann_path) else None
                item_info_path = item_info_path if os.path.isfile(item_info_path) else None
                try:
                    new_ds.add_item_file(
                        item_name,
                        item_path,
                        ann_path,
                        _validate_item=_validate_item,
                        _use_hardlink=_use_hardlink,
                        item_info=item_info_path,
                    )
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "source_project_name": self.name,
                            "dst_directory": dst_directory,
                            "ds_name": ds.name,
                            "item_name": item_name,
                            "item_path": item_path,
                            "ann_path": ann_path,
                            "item_info": item_info_path,
                        },
                    )
                    raise e
        new_project.set_key_id_map(self.key_id_map)
        return new_project

    @staticmethod
    def to_segmentation_task(
        src_project_dir: str,
        dst_project_dir: Optional[str] = None,
        inplace: Optional[bool] = False,
        target_classes: Optional[List[str]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        segmentation_type: Optional[str] = "semantic",
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'to_segmentation_task()' is not supported for VideoProject class now."
        )

    @staticmethod
    def to_detection_task(
        src_project_dir: str,
        dst_project_dir: Optional[str] = None,
        inplace: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'to_detection_task()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_classes_except(
        project_dir: str,
        classes_to_keep: Optional[List[str]] = None,
        inplace: Optional[bool] = False,
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_classes_except()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_classes(
        project_dir: str,
        classes_to_remove: Optional[List[str]] = None,
        inplace: Optional[bool] = False,
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_classes()' is not supported for VideoProject class now."
        )

    @staticmethod
    def _remove_items(
        project_dir,
        without_objects=False,
        without_tags=False,
        without_objects_and_tags=False,
        inplace=False,
    ):
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method '_remove_items()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_items_without_objects(project_dir: str, inplace: Optional[bool] = False) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_items_without_objects()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_items_without_tags(project_dir: str, inplace: Optional[bool] = False) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_items_without_tags()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_items_without_both_objects_and_tags(
        project_dir: str, inplace: Optional[bool] = False
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_items_without_both_objects_and_tags()' is not supported for VideoProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_count(project_dir: str, train_count: int, val_count: int) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_count()' is not supported for VideoProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_tag(
        project_dir: str,
        train_tag_name: str,
        val_tag_name: str,
        untagged: Optional[str] = "ignore",
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_tag()' is not supported for VideoProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_dataset(
        project_dir: str, train_datasets: List[str], val_datasets: List[str]
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_tag()' is not supported for VideoProject class now."
        )

    @classmethod
    def read_single(cls, dir):
        """
        Read project from given ditectory. Generate exception error if given dir contains more than one subdirectory
        :param dir: str
        :return: VideoProject class object
        """
        return read_project_wrapper(dir, cls)

    @staticmethod
    def download(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: List[int] = None,
        download_videos: bool = True,
        save_video_info: bool = False,
        log_progress: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download video project from Supervisely to the given directory.

        :param api: Supervisely Api class object.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Project ID in Supervisely.
        :type project_id: :class:`int`
        :param dest_dir: Directory to download video project.
        :type dest_dir: :class:`str`
        :param dataset_ids: Datasets IDs in Supervisely to download.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param download_videos: Download videos from Supervisely video project in dest_dir or not.
        :type download_videos: :class:`bool`, optional
        :param save_video_info: Save video infos or not.
        :type save_video_info: :class:`bool`, optional
        :param log_progress: Log download progress or not.
        :type log_progress: :class:`bool`, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: :class:`tqdm`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local destination Project folder
            save_directory = "/home/admin/work/supervisely/source/video_project"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)
            project_id = 8888

            # Download Video Project
            sly.VideoProject.download(api, project_id, save_directory)
            project_fs = sly.VideoProject(save_directory, sly.OpenMode.READ)
        """
        download_video_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            download_videos=download_videos,
            save_video_info=save_video_info,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    def upload(
        dir: str,
        api: Api,
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: Optional[bool] = True,
    ) -> Tuple[int, str]:
        """
        Upload video project from given directory in Supervisely.

        :param dir: Directory with video project.
        :type dir: str
        :param api: Api class object.
        :type api: Api
        :param workspace_id: Workspace ID in Supervisely to upload video project.
        :type workspace_id: int
        :param project_name: Name of video project.

        :type project_name: str
        :param log_progress: Logging progress of download video project or not.
        :type log_progress: bool, optional
        :return: New video project ID in Supervisely and project name
        :rtype: :class:`int`, :class:`str`
        :Usage example:

        .. code-block:: python

                import supervisely as sly

                # Local folder with Video Project
                project_directory = "/home/admin/work/supervisely/source/video_project"

                # Obtain server address and your api_token from environment variables
                # Edit those values if you run this notebook on your own PC
                address = os.environ['SERVER_ADDRESS']
                token = os.environ['API_TOKEN']

                # Initialize API object
                api = sly.Api(address, token)

                # Upload Video Project
                project_id, project_name = sly.VideoProject.upload(
                    project_directory,
                    api,
                    workspace_id=45,
                    project_name="My Video Project"
                )
        """
        return upload_video_project(
            dir=dir,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
        )


def download_video_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_videos: Optional[bool] = True,
    save_video_info: Optional[bool] = False,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    include_custom_data: Optional[bool] = False,
) -> None:
    """
    Download video project to the local directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID to download
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param download_videos: Include videos in the download.
    :type download_videos: bool, optional
    :param save_video_info: Include video info in the download.
    :type save_video_info: bool, optional
    :param log_progress: Show downloading logs in the output.
    :type log_progress: bool, optional
    :param progress_cb: Function for tracking download progress.
    :type progress_cb: tqdm or callable, optional

    :return: None.
    :rtype: NoneType
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        from tqdm import tqdm
        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        dest_dir = 'your/local/dest/dir'

        # Download video project
        project_id = 17758
        project_info = api.project.get_info_by_id(project_id)
        num_videos = project_info.items_count

        p = tqdm(desc="Downloading video project", total=num_videos)
        sly.download(
            api,
            project_id,
            dest_dir,
            progress_cb=p,
        )
    """
    LOG_BATCH_SIZE = 1

    key_id_map = KeyIdMap()

    project_fs = VideoProject(dest_dir, OpenMode.CREATE)

    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    datasets_infos = []
    if dataset_ids is not None:
        for ds_id in dataset_ids:
            datasets_infos.append(api.dataset.get_info_by_id(ds_id))
    else:
        datasets_infos = api.dataset.get_list(project_id)

    for dataset in datasets_infos:
        dataset_fs = project_fs.create_dataset(dataset.name)
        videos = api.video.get_list(dataset.id)

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset.name), total_cnt=len(videos)
            )
        for batch in batched(videos, batch_size=LOG_BATCH_SIZE):
            video_ids = [video_info.id for video_info in batch]
            video_names = [video_info.name for video_info in batch]
            custom_datas = [video_info.custom_data for video_info in batch]

            try:
                ann_jsons = api.video.annotation.download_bulk(dataset.id, video_ids)
            except Exception as e:
                logger.info(
                    "INFO FOR DEBUGGING",
                    extra={
                        "project_id": project_id,
                        "dataset_id": dataset.id,
                        "video_ids": video_ids,
                    },
                )
                raise e
            for video_id, video_name, custom_data, ann_json, video_info in zip(
                video_ids, video_names, custom_datas, ann_jsons, batch
            ):
                if video_name != ann_json[ApiField.VIDEO_NAME]:
                    raise RuntimeError("Error in api.video.annotation.download_batch: broken order")

                video_file_path = dataset_fs.generate_item_path(video_name)

                if include_custom_data:
                    CUSTOM_DATA_DIR = os.path.join(dest_dir, dataset.name, "custom_data")
                    mkdir(CUSTOM_DATA_DIR)
                    custom_data_path = os.path.join(CUSTOM_DATA_DIR, f"{video_name}.json")
                    dump_json_file(custom_data, custom_data_path)

                if download_videos:
                    try:
                        video_file_size = video_info.file_meta.get("size")
                        if log_progress and video_file_size is not None:
                            item_progress = Progress(
                                f"Downloading {video_name}",
                                total_cnt=int(video_file_size),
                                is_size=True,
                            )
                            api.video.download_path(
                                video_id, video_file_path, item_progress.iters_done_report
                            )
                        else:
                            api.video.download_path(video_id, video_file_path)
                    except Exception as e:
                        logger.info(
                            "INFO FOR DEBUGGING",
                            extra={
                                "project_id": project_id,
                                "dataset_id": dataset.id,
                                "video_id": video_id,
                                "video_file_path": video_file_path,
                            },
                        )
                        raise e
                else:
                    touch(video_file_path)
                item_info = video_info._asdict() if save_video_info else None
                try:
                    video_ann = VideoAnnotation.from_json(ann_json, project_fs.meta, key_id_map)
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "project_id": project_id,
                            "dataset_id": dataset.id,
                            "video_id": video_id,
                            "video_name": video_name,
                            "ann_json": ann_json,
                        },
                    )
                    raise e
                try:
                    dataset_fs.add_item_file(
                        video_name,
                        video_file_path,
                        ann=video_ann,
                        _validate_item=False,
                        _use_hardlink=True,
                        item_info=item_info,
                    )
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "project_id": project_id,
                            "dataset_id": dataset.id,
                            "video_id": video_id,
                            "video_name": video_name,
                            "video_file_path": video_file_path,
                            "item_info": item_info,
                        },
                    )
                    raise e

                if progress_cb is not None:
                    progress_cb(1)

            if log_progress:
                ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)


def upload_video_project(
    dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: Optional[bool] = True,
    include_custom_data: Optional[bool] = False,
) -> Tuple[int, str]:
    project_fs = VideoProject.read_single(dir)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.VIDEOS)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    for dataset_fs in project_fs.datasets:
        dataset = api.dataset.create(project.id, dataset_fs.name)

        names, item_paths, ann_paths = [], [], []
        for item_name in dataset_fs:
            video_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            item_paths.append(video_path)
            ann_paths.append(ann_path)

        progress_cb = None
        if log_progress:
            ds_progress = Progress(
                "Uploading videos to dataset {!r}".format(dataset.name),
                total_cnt=len(item_paths),
            )
            progress_cb = ds_progress.iters_done_report
        try:
            item_infos = api.video.upload_paths(dataset.id, names, item_paths, progress_cb)

            if include_custom_data:
                for item_info in item_infos:
                    item_name = item_info.name
                    custom_data_path = os.path.join(
                        dir, dataset_fs.name, "custom_data", f"{item_name}.json"
                    )

                    if os.path.exists(custom_data_path):
                        custom_data = load_json_file(custom_data_path)
                        api.video.update_custom_data(item_info.id, custom_data)

        except Exception as e:
            logger.info(
                "INFO FOR DEBUGGING",
                extra={
                    "project_id": project.id,
                    "dataset_id": dataset.id,
                    "names": names,
                    "item_paths": item_paths,
                },
            )
            raise e
        item_ids = [item_info.id for item_info in item_infos]
        if log_progress:
            ds_progress = Progress(
                "Uploading annotations to dataset {!r}".format(dataset.name),
                total_cnt=len(item_paths),
            )
            progress_cb = ds_progress.iters_done_report
        try:
            api.video.annotation.upload_paths(item_ids, ann_paths, project_fs.meta, progress_cb)
        except Exception as e:
            logger.info(
                "INFO FOR DEBUGGING",
                extra={
                    "project_id": project.id,
                    "dataset_id": dataset.id,
                    "item_ids": item_ids,
                    "ann_paths": ann_paths,
                },
            )
            raise e

    return project.id, project.name
