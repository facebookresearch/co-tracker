# coding: utf-8

# docs
from __future__ import annotations
import os
import random
from supervisely.api.api import Api
from typing import Tuple, List, Dict, Optional, Callable, NamedTuple, Union

from tqdm import tqdm
import supervisely.imaging.image as sly_image
from supervisely._utils import batched
from supervisely.api.module_api import ApiField
from supervisely.api.pointcloud.pointcloud_api import PointcloudInfo
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.io.fs import touch, dir_exists, list_files, mkdir, get_file_name
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import (
    PointcloudEpisodeAnnotation,
)
from supervisely.project.pointcloud_project import PointcloudProject, PointcloudDataset
from supervisely.project.project import OpenMode
from supervisely.project.project import read_single_project as read_project_wrapper
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.frame import Frame
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger


class EpisodeItemPaths(NamedTuple):
    #: :class:`str`: Full pointcloud file path of item
    pointcloud_path: str

    #: :class:`str`: Path to related images directory of item
    related_images_dir: str

    #: :class:`int`: Index of frame in episode annotation of dataset
    frame_index: int


class EpisodeItemInfo(NamedTuple):
    #: :class:`str`: Item's dataset name
    dataset_name: str

    #: :class:`str`: Item name
    name: str

    #: :class:`str`: Full pointcloud file path of item
    pointcloud_path: str

    #: :class:`str`: Path to related images directory of item
    related_images_dir: str

    #: :class:`int`: Index of frame in episode annotation of dataset
    frame_index: int


class PointcloudEpisodeDataset(PointcloudDataset):
    #: :class:`str`: Items data directory name
    item_dir_name = "pointcloud"

    #: :class:`str`: Annotations directory name
    ann_dir_name = "ann"

    #: :class:`str`: Items info directory name
    item_info_dir_name = "pointcloud_info"

    #: :class:`str`: Related images directory name
    related_images_dir_name = "related_images"

    #: :class:`str`: Segmentation masks directory name
    seg_dir_name = None

    item_info_type = PointcloudInfo
    annotation_class = PointcloudEpisodeAnnotation

    @property
    def ann_dir(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} object don't have correct path for 'ann_dir' property. \
            Use 'get_ann_path()' method instead of this."
        )

    def get_item_paths(self, item_name: str) -> EpisodeItemPaths:
        return EpisodeItemPaths(
            pointcloud_path=self.get_pointcloud_path(item_name),
            related_images_dir=self.get_related_images_path(item_name),
            frame_index=self.get_frame_idx(item_name),
        )

    def get_ann_path(self) -> str:
        return os.path.join(self.directory, "annotation.json")

    def get_ann(
        self, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudEpisodeAnnotation:
        """
        Read pointcloud annotation of item from json.

        :param item_name: Pointcloud name.
        :type item_name: str
        :param project_meta: Project Meta.
        :type project_meta: :class:`ProjectMeta<supervisely.ProjectMeta>`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: PointcloudEpisodeAnnotation object
        :rtype: :class:`PointcloudEpisodeAnnotation<supervisely.PointcloudEpisodeAnnotation>`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = sly.PointcloudProject(project_path, sly.OpenMode.READ)

            ds = project.datasets.get('ds1')

            annotation = ds.get_ann("PTC_0056")
            # Output: RuntimeError: Item PTC_0056 not found in the project.

            annotation = ds.get_ann("PTC_0056.pcd")
            print(type(annotation).__name__)
            # Output: PointcloudEpisodeAnnotation
        """
        ann_path = self.get_ann_path()
        return self.annotation_class.load_json_file(ann_path, project_meta, key_id_map)

    def get_ann_frame(self, item_name: str, annotation: PointcloudEpisodeAnnotation) -> Frame:
        frame_idx = self.get_frame_idx(item_name)
        if frame_idx is None:
            raise ValueError(f"Frame wasn't assigned to pointcloud with name {item_name}.")
        return annotation.frames.get(frame_idx)

    def get_frame_pointcloud_map_path(self) -> str:
        return os.path.join(self.directory, "frame_pointcloud_map.json")

    def set_ann(self, ann: PointcloudEpisodeAnnotation) -> None:
        if type(ann) is not self.annotation_class:
            raise TypeError(
                f"Type of 'ann' should be {self.annotation_class.__name__}, not a {type(ann).__name__}"
            )
        dst_ann_path = self.get_ann_path()
        dump_json_file(ann.to_json(), dst_ann_path)

    def _create(self):
        mkdir(self.item_dir)

    def _read(self):
        if not dir_exists(self.item_dir):
            raise NotADirectoryError(
                f"Cannot read dataset {self.name}: {self.item_dir} directory not found"
            )

        try:
            item_paths = sorted(list_files(self.item_dir, filter_fn=self._has_valid_ext))
            item_names = sorted([os.path.basename(path) for path in item_paths])

            map_file_path = self.get_frame_pointcloud_map_path()
            if os.path.isfile(map_file_path):
                self._frame_to_pc_map = load_json_file(map_file_path)
            else:
                self._frame_to_pc_map = {
                    frame_index: item_names[frame_index] for frame_index in range(len(item_names))
                }

            self._pc_to_frame = {v: k for k, v in self._frame_to_pc_map.items()}
            self._item_to_ann = {name: self._pc_to_frame[name] for name in item_names}
        except Exception as ex:
            raise Exception(f"Cannot read dataset ({self.name}): {repr(ex)}")

    def add_item_file(
        self,
        item_name: str,
        item_path: str,
        frame: Optional[Union[str, int]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        item_info: Optional[NamedTuple] = None,
    ) -> None:
        """
        Adds given item file to dataset items directory, and adds given annotation to dataset ann directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: str
        :param item_path: Path to the item.
        :type item_path: str
        :param frame: Frame number.
        :type frame: str or int, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: bool, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: bool, optional
        :param item_info: NamedTuple ImageInfo containing information about pointcloud.
        :type item_info: NamedTuple, optional
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`Exception` if item_name already exists in dataset or item name has unsupported extension.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/episodes_project/episode_0"
            ds = sly.PointcloudEpisodeDataset(dataset_path, sly.OpenMode.READ)

            ds.add_item_file("PTC_777.pcd", "/home/admin/work/supervisely/projects/episodes_project/episode_0/pointcloud/PTC_777.pcd", frame=3)
        """
        if item_path is None and item_info is None:
            raise RuntimeError("No item_path or ann or item_info provided.")

        self._add_item_file(
            item_name,
            item_path,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
        )
        self._add_ann_by_type(item_name, frame)
        self._add_item_info(item_name, item_info)

    def get_classes_stats(
        self,
        project_meta: Optional[ProjectMeta] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        if project_meta is None:
            project = PointcloudEpisodeProject(self.project_dir, OpenMode.READ)
            project_meta = project.meta
        class_items = {}
        class_objects = {}
        class_figures = {}
        for obj_class in project_meta.obj_classes:
            class_items[obj_class.name] = 0
            class_objects[obj_class.name] = 0
            class_figures[obj_class.name] = 0
        episode_ann: PointcloudEpisodeAnnotation = self.get_ann(project_meta)
        for ann_obj in episode_ann.objects:
            class_objects[ann_obj.obj_class.name] += 1
        for item_name in self:
            frame_index = self.get_frame_idx(item_name)
            item_figures = episode_ann.get_figures_on_frame(frame_index)
            item_class = {}
            for ptc_figure in item_figures:
                class_figures[ptc_figure.parent_object.obj_class.name] += 1
                item_class[ptc_figure.parent_object.obj_class.name] = True
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

    def _add_ann_by_type(self, item_name, frame):
        if frame is None:
            self._item_to_ann[item_name] = ""
        elif isinstance(frame, int):
            self._item_to_ann[item_name] = str(frame)
        elif type(frame) is str:
            self._item_to_ann[item_name] = frame
        else:
            raise TypeError("Unsupported type {!r} for ann argument".format(type(frame)))

    def get_frame_idx(self, item_name: str) -> int:
        frame = self._item_to_ann.get(item_name, None)
        if frame is None:
            raise RuntimeError("Item {} not found in the project.".format(item_name))
        if self._item_to_ann[item_name] == "":
            return None
        return int(self._item_to_ann[item_name])


class PointcloudEpisodeProject(PointcloudProject):
    dataset_class = PointcloudEpisodeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = PointcloudEpisodeDataset

    @classmethod
    def read_single(cls, dir) -> PointcloudEpisodeProject:
        return read_project_wrapper(dir, cls)

    def get_classes_stats(
        self,
        dataset_names: Optional[List[str]] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        return super(PointcloudEpisodeProject, self).get_classes_stats(
            dataset_names, return_objects_count, return_figures_count, return_items_count
        )

    @staticmethod
    def get_train_val_splits_by_count(
        project_dir: str, train_count: int, val_count: int
    ) -> Tuple[List[EpisodeItemInfo], List[EpisodeItemInfo]]:
        """
        Get train and val items information from project by given train and val counts.

        :param project_dir: Path to project directory.
        :type project_dir: str
        :param train_count: Number of train items.
        :type train_count: int
        :param val_count: Number of val items.
        :type val_count: int
        :raises: :class:`ValueError` if total_count != train_count + val_count
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`Tuple[List[EpisodeItemInfo], List[EpisodeItemInfo]]`
        :Usage example:

         .. code-block:: python

            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = sly.PointcloudEpisodeProject(project_path, sly.OpenMode.READ)
            train_count = 16
            val_count = 4
            train_items, val_items = project.get_train_val_splits_by_count(project_path, train_count, val_count)
        """

        def _list_items_for_splits(project) -> List[EpisodeItemInfo]:
            items = []
            for dataset in project.datasets:
                for item_name in dataset:
                    items.append(
                        EpisodeItemInfo(
                            dataset_name=dataset.name,
                            name=item_name,
                            pointcloud_path=dataset.get_pointcloud_path(item_name),
                            related_images_dir=dataset.get_related_images_path(item_name),
                            frame_index=dataset.get_frame_idx(item_name),
                        )
                    )
            return items

        project = PointcloudEpisodeProject(project_dir, OpenMode.READ)
        if project.total_items != train_count + val_count:
            raise ValueError("total_count != train_count + val_count")
        all_items = _list_items_for_splits(project)
        random.shuffle(all_items)
        train_items = all_items[:train_count]
        val_items = all_items[train_count:]
        return train_items, val_items

    @staticmethod
    def get_train_val_splits_by_tag(
        project_dir: str,
        train_tag_name: str,
        val_tag_name: str,
        untagged: Optional[str] = "ignore",
    ) -> Tuple[List[EpisodeItemInfo], List[EpisodeItemInfo]]:
        """
        Get train and val items information from project by given train and val tags names.

        :param project_dir: Path to project directory.
        :type project_dir: str
        :param train_tag_name: Train tag name.
        :type train_tag_name: str
        :param val_tag_name: Val tag name.
        :type val_tag_name: str
        :param untagged: Actions in case of absence of train_tag_name and val_tag_name in project.
        :type untagged: str, optional
        :raises: :class:`ValueError` if untagged not in ["ignore", "train", "val"]
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`Tuple[List[EpisodeItemInfo], List[EpisodeItemInfo]]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = sly.PointcloudEpisodeProject(project_path, sly.OpenMode.READ)
            train_tag_name = 'train'
            val_tag_name = 'val'
            train_items, val_items = project.get_train_val_splits_by_tag(project_path, train_tag_name, val_tag_name)
        """
        untagged_actions = ["ignore", "train", "val"]
        if untagged not in untagged_actions:
            raise ValueError(
                f"Unknown untagged action {untagged}. Should be one of {untagged_actions}"
            )
        project = PointcloudEpisodeProject(project_dir, OpenMode.READ)
        train_items = []
        val_items = []
        for dataset in project.datasets:
            ann = dataset.get_ann(project.meta)
            for item_name in dataset:
                item_paths = dataset.get_item_paths(item_name)
                frame_idx = dataset.get_frame_idx(item_name)

                info = EpisodeItemInfo(
                    dataset_name=dataset.name,
                    name=item_name,
                    pointcloud_path=item_paths.pointcloud_path,
                    related_images_dir=item_paths.related_images_dir,
                    frame_index=frame_idx,
                )
                frame_tags = ann.get_tags_on_frame(frame_idx)
                if frame_tags.get(train_tag_name) is not None:
                    train_items.append(info)
                if frame_tags.get(val_tag_name) is not None:
                    val_items.append(info)
                if frame_tags.get(train_tag_name) is None and frame_tags.get(val_tag_name) is None:
                    # untagged item
                    if untagged == "ignore":
                        continue
                    elif untagged == "train":
                        train_items.append(info)
                    elif untagged == "val":
                        val_items.append(info)
        return train_items, val_items

    @staticmethod
    def get_train_val_splits_by_dataset(
        project_dir: str, train_datasets: List[str], val_datasets: List[str]
    ) -> Tuple[List[EpisodeItemInfo], List[EpisodeItemInfo]]:
        """
        Get train and val items information from project by given train and val datasets names.

        :param project_dir: Path to project directory.
        :type project_dir: str
        :param train_datasets: List of train datasets names.
        :type train_datasets: List[str]
        :param val_datasets: List of val datasets names.
        :type val_datasets: List[str]
        :raises: :class:`KeyError` if dataset name not found in project
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`Tuple[List[EpisodeItemInfo], List[EpisodeItemInfo]]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = sly.PointcloudEpisodeProject(project_path, sly.OpenMode.READ)
            train_datasets = ['ds1', 'ds2']
            val_datasets = ['ds3', 'ds4']
            train_items, val_items = project.get_train_val_splits_by_dataset(project_path, train_datasets, val_datasets)
        """

        def _add_items_to_list(project, datasets_names, items_list):
            for dataset_name in datasets_names:
                dataset = project.datasets.get(dataset_name)
                if dataset is None:
                    raise KeyError(f"Dataset '{dataset_name}' not found")
                for item_name in dataset:
                    item_paths = dataset.get_item_paths(item_name)
                    frame_idx = dataset.get_frame_idx(item_name)
                    info = EpisodeItemInfo(
                        dataset_name=dataset.name,
                        name=item_name,
                        pointcloud_path=item_paths.pointcloud_path,
                        related_images_dir=item_paths.related_images_dir,
                        frame_index=frame_idx,
                    )
                    items_list.append(info)

        project = PointcloudEpisodeProject(project_dir, OpenMode.READ)
        train_items = []
        _add_items_to_list(project, train_datasets, train_items)
        val_items = []
        _add_items_to_list(project, val_datasets, val_items)
        return train_items, val_items

    @staticmethod
    def download(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: Optional[List[int]] = None,
        download_pointclouds: Optional[bool] = True,
        download_related_images: Optional[bool] = True,
        download_pointclouds_info: Optional[bool] = False,
        batch_size: Optional[int] = 10,
        log_progress: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download pointcloud episodes project from Supervisely to the given directory.

        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Supervisely downloadable project ID.
        :type project_id: :class:`int`
        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`
        :param dataset_ids: Dataset IDs.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param download_pointclouds: Download pointcloud data files or not.
        :type download_pointclouds: :class:`bool`, optional
        :param download_related_images: Download related images or not.
        :type download_related_images: :class:`bool`, optional
        :param download_pointclouds_info: Download pointcloud info .json files or not.
        :type download_pointclouds_info: :class:`bool`, optional
        :param batch_size: The number of images in the batch when they are loaded to a host.
        :type batch_size: :class:`int`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: :class:`tqdm` or callable, optional
        :return: None
        :rtype: NoneType
        :Usage example:

        .. code-block:: python

                import supervisely as sly

                # Local destination project folder
                save_directory = "/home/admin/work/supervisely/source/ptc_project"

                # Obtain server address and your api_token from environment variables
                # Edit those values if you run this notebook on your own PC
                address = os.environ['SERVER_ADDRESS']
                token = os.environ['API_TOKEN']

                # Initialize API object
                api = sly.Api(address, token)
                project_id = 8888

                # Download Project
                sly.PointcloudEpisodeProject.download(api, project_id, save_directory)
                project_fs = sly.PointcloudEpisodeProject(save_directory, sly.OpenMode.READ)
        """
        download_pointcloud_episode_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            download_pcd=download_pointclouds,
            download_related_images=download_related_images,
            download_annotations=True,
            download_pointclouds_info=download_pointclouds_info,
            batch_size=batch_size,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    def upload(
        directory: str,
        api: Api,
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> Tuple[int, str]:
        """
        Uploads pointcloud episodes project to Supervisely from the given directory.

        :param directory: Path to project directory.
        :type directory: :class:`str`
        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param workspace_id: Workspace ID, where project will be uploaded.
        :type workspace_id: :class:`int`
        :param project_name: Name of the project in Supervisely. Can be changed if project with the same name is already exists.
        :type project_name: :class:`str`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: :class:`tqdm` or callable, optional
        :return: Project ID and name. It is recommended to check that returned project name coincides with provided project name.
        :rtype: :class:`int`, :class:`str`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local folder with Pointcloud Project
            project_directory = "/home/admin/work/supervisely/source/episodes_project"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)

            # Upload Pointcloud Project
            project_id, project_name = sly.PointcloudEpisodeProject.upload(
                project_directory,
                api,
                workspace_id=45,
                project_name="My Episodes Project"
            )
        """
        return upload_pointcloud_episode_project(
            directory=directory,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )


def download_pointcloud_episode_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_pcd: Optional[bool] = True,
    download_related_images: Optional[bool] = True,
    download_annotations: Optional[bool] = True,
    download_pointclouds_info: Optional[bool] = False,
    batch_size: Optional[int] = 10,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """
    Download pointcloud episode project to the local directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID to download.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param download_pcd: Include pointcloud episode items in the download.
    :type download_pcd: bool, optional
    :param download_related_images: Include related context images in the download.
    :type download_related_images: bool, optional
    :param download_annotations: Include annotations in the download.
    :type download_annotations: bool, optional
    :param download_pointclouds_info: Include pointclouds info in the download.
    :type download_pointclouds_info: bool, optional
    :param batch_size: Size of a downloading batch.
    :type batch_size: int, optional
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

        # Download pointcloud episodes project
        project_id = 19636
        project_info = api.project.get_info_by_id(project_id)
        num_pointclouds_ep = project_info.items_count

        p = tqdm(desc="Downloading pointcloud project", total=num_pointclouds_ep)
        sly.download_pointcloud_project(
            api,
            project_id,
            dest_dir,
            progress_cb=p,
        )
    """

    key_id_map = KeyIdMap()
    project_fs = PointcloudEpisodeProject(dest_dir, OpenMode.CREATE)
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
        pointclouds = api.pointcloud_episode.get_list(dataset.id)

        # Download annotation to project_path/dataset_path/annotation.json
        if download_annotations is True:
            ann_json = api.pointcloud_episode.annotation.download(dataset.id)
            annotation = dataset_fs.annotation_class.from_json(ann_json, meta, key_id_map)
            dataset_fs.set_ann(annotation)

        # frames --> pointcloud mapping to project_path/dataset_path/frame_pointcloud_map.json
        frame_name_map = api.pointcloud_episode.get_frame_name_map(dataset.id)
        frame_pointcloud_map_path = dataset_fs.get_frame_pointcloud_map_path()
        dump_json_file(frame_name_map, frame_pointcloud_map_path)

        # Download data
        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset.name), total_cnt=len(pointclouds)
            )

        for batch in batched(pointclouds, batch_size=batch_size):
            pointcloud_ids = [pointcloud_info.id for pointcloud_info in batch]
            pointcloud_names = [pointcloud_info.name for pointcloud_info in batch]
            map_file_path = dataset_fs.get_frame_pointcloud_map_path()
            frame_to_pc_map = load_json_file(map_file_path)
            pc_to_frame = {v: k for k, v in frame_to_pc_map.items()}
            item_to_ann = {name: pc_to_frame[name] for name in pointcloud_names}

            for pointcloud_id, pointcloud_name, pointcloud_info in zip(
                pointcloud_ids, pointcloud_names, batch
            ):
                pointcloud_file_path = dataset_fs.generate_item_path(pointcloud_name)
                if download_pcd:
                    try:
                        api.pointcloud_episode.download_path(pointcloud_id, pointcloud_file_path)
                    except Exception as e:
                        logger.info(
                            "INFO FOR DEBUGGING",
                            extra={
                                "project_id": project_id,
                                "dataset_id": dataset.id,
                                "pointcloud_id": pointcloud_id,
                                "pointcloud_name": pointcloud_name,
                                "pointcloud_file_path": pointcloud_file_path,
                            },
                        )
                        raise e
                else:
                    touch(pointcloud_file_path)

                if download_related_images:
                    related_images_path = dataset_fs.get_related_images_path(pointcloud_name)
                    try:
                        related_images = api.pointcloud_episode.get_list_related_images(
                            pointcloud_id
                        )
                    except Exception as e:
                        logger.info(
                            "INFO FOR DEBUGGING",
                            extra={
                                "project_id": project_id,
                                "dataset_id": dataset.id,
                                "pointcloud_id": pointcloud_id,
                                "pointcloud_name": pointcloud_name,
                            },
                        )
                        raise e

                    for rimage_info in related_images:
                        name = rimage_info[ApiField.NAME]
                        if not sly_image.has_valid_ext(name):
                            new_name = get_file_name(name)  # to fix cases like .png.json
                            if sly_image.has_valid_ext(new_name):
                                name = new_name
                                rimage_info[ApiField.NAME] = name
                            else:
                                raise RuntimeError(
                                    "Something wrong with photo context filenames.\
                                                    Please, contact support"
                                )
                        rimage_id = rimage_info[ApiField.ID]

                        path_img = os.path.join(related_images_path, name)
                        path_json = os.path.join(related_images_path, name + ".json")

                        try:
                            api.pointcloud_episode.download_related_image(rimage_id, path_img)
                        except Exception as e:
                            logger.info(
                                "INFO FOR DEBUGGING",
                                extra={
                                    "project_id": project_id,
                                    "dataset_id": dataset.id,
                                    "pointcloud_id": pointcloud_id,
                                    "pointcloud_name": pointcloud_name,
                                    "rimage_id": rimage_id,
                                    "path_img": path_img,
                                },
                            )
                            raise e

                        dump_json_file(rimage_info, path_json)

                pointcloud_info = pointcloud_info._asdict() if download_pointclouds_info else None
                try:
                    dataset_fs.add_item_file(
                        pointcloud_name,
                        pointcloud_file_path,
                        item_to_ann[pointcloud_name],
                        _validate_item=False,
                        item_info=pointcloud_info,
                    )
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "project_id": project_id,
                            "dataset_id": dataset.id,
                            "pointcloud_id": pointcloud_id,
                            "pointcloud_name": pointcloud_name,
                            "pointcloud_file_path": pointcloud_file_path,
                            "item_info": pointcloud_info,
                        },
                    )
                    raise e

                if progress_cb is not None:
                    progress_cb(1)
            if log_progress:
                ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)


def upload_pointcloud_episode_project(
    directory: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> Tuple[int, str]:
    # STEP 0 — create project remotely
    project_locally = PointcloudEpisodeProject.read_single(directory)
    project_name = project_locally.name if project_name is None else project_name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project_remotely = api.project.create(
        workspace_id, project_name, ProjectType.POINT_CLOUD_EPISODES
    )
    api.project.update_meta(project_remotely.id, project_locally.meta.to_json())

    key_id_map = KeyIdMap()
    for dataset_locally in project_locally.datasets:
        ann_json_path = dataset_locally.get_ann_path()

        if os.path.isfile(ann_json_path):
            ann_json = load_json_file(ann_json_path)
            episode_annotation = PointcloudEpisodeAnnotation.from_json(
                ann_json, project_locally.meta
            )
        else:
            episode_annotation = PointcloudEpisodeAnnotation()

        dataset_remotely = api.dataset.create(
            project_remotely.id,
            dataset_locally.name,
            description=episode_annotation.description,
            change_name_if_conflict=True,
        )

        # STEP 1 — upload episodes
        items_infos = {"names": [], "paths": [], "metas": []}

        for item_name in dataset_locally:
            item_path, related_images_dir, frame_idx = dataset_locally.get_item_paths(item_name)

            item_meta = {"frame": frame_idx}

            items_infos["names"].append(item_name)
            items_infos["paths"].append(item_path)
            items_infos["metas"].append(item_meta)

        progress = None
        if log_progress and progress_cb is None:
            progress = Progress(
                "Uploading pointclouds: {!r}".format(dataset_remotely.name),
                total_cnt=len(dataset_locally),
            ).iters_done_report
        elif progress_cb is not None:
            progress = progress_cb
        try:
            pcl_infos = api.pointcloud_episode.upload_paths(
                dataset_remotely.id,
                names=items_infos["names"],
                paths=items_infos["paths"],
                metas=items_infos["metas"],
                progress_cb=progress,
            )
        except Exception as e:
            logger.info(
                "INFO FOR DEBUGGING",
                extra={
                    "project_id": project_remotely.id,
                    "dataset_id": dataset_remotely.id,
                    "item_names": items_infos["names"],
                    "item_paths": items_infos["paths"],
                    "item_metas": items_infos["metas"],
                },
            )
            raise e
        # STEP 2 — upload annotations
        frame_to_pcl_ids = {pcl_info.frame: pcl_info.id for pcl_info in pcl_infos}
        try:
            api.pointcloud_episode.annotation.append(
                dataset_remotely.id, episode_annotation, frame_to_pcl_ids, key_id_map
            )
        except Exception as e:
            logger.info(
                "INFO FOR DEBUGGING",
                extra={
                    "project_id": project_remotely.id,
                    "dataset_id": dataset_remotely.id,
                    "frame_to_pcl_ids": frame_to_pcl_ids,
                    "ann": episode_annotation.to_json(),
                },
            )
            raise e

        # STEP 3 — upload photo context
        img_infos = {"img_paths": [], "img_metas": []}

        # STEP 3.1 — upload images
        for pcl_info in pcl_infos:
            related_items = dataset_locally.get_related_images(pcl_info.name)
            images_paths_for_frame = [img_path for img_path, _ in related_items]

            img_infos["img_paths"].extend(images_paths_for_frame)

        if log_progress and progress_cb is None:
            progress = Progress(
                "Uploading photo context: {!r}".format(dataset_remotely.name),
                total_cnt=len(img_infos["img_paths"]),
            ).iters_done_report
        elif progress_cb is not None:
            progress = progress_cb

        try:
            images_hashes = api.pointcloud_episode.upload_related_images(
                img_infos["img_paths"],
                progress_cb=progress,
            )
        except Exception as e:
            logger.info(
                "INFO FOR DEBUGGING",
                extra={
                    "project_id": project_remotely.id,
                    "dataset_id": dataset_remotely.id,
                    "img_paths": img_infos["img_paths"],
                },
            )
            raise e

        # STEP 3.2 — upload images metas
        images_hashes_iterator = images_hashes.__iter__()
        for pcl_info in pcl_infos:
            related_items = dataset_locally.get_related_images(pcl_info.name)

            for img_ind, (_, meta_json) in enumerate(related_items):
                img_hash = next(images_hashes_iterator)
                if "deviceId" not in meta_json[ApiField.META].keys():
                    meta_json[ApiField.META]["deviceId"] = f"CAM_{str(img_ind).zfill(2)}"
                img_infos["img_metas"].append(
                    {
                        ApiField.ENTITY_ID: pcl_info.id,
                        ApiField.NAME: meta_json[ApiField.NAME],
                        ApiField.HASH: img_hash,
                        ApiField.META: meta_json[ApiField.META],
                    }
                )

        if len(img_infos["img_metas"]) > 0:
            try:
                api.pointcloud_episode.add_related_images(img_infos["img_metas"])
            except Exception as e:
                logger.info(
                    "INFO FOR DEBUGGING",
                    extra={
                        "project_id": project_remotely.id,
                        "dataset_id": dataset_remotely.id,
                        "rimg_infos": img_infos["img_metas"],
                    },
                )
                raise e

    return project_remotely.id, project_remotely.name
