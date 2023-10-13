# coding: utf-8

from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Callable, NamedTuple, Union

from supervisely.api.api import Api
from collections import namedtuple
import os
import random
import numpy as np
import shutil
from tqdm import tqdm

from supervisely.io.fs import (
    file_exists,
    touch,
    dir_exists,
    list_files,
    get_file_name_with_ext,
    get_file_name,
    copy_file,
    silent_remove,
    remove_dir,
    ensure_base_path,
)
import supervisely.imaging.image as sly_image
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely._utils import batched
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.api.module_api import ApiField
from supervisely.api.pointcloud.pointcloud_api import PointcloudInfo
from supervisely.collection.key_indexed_collection import KeyIndexedCollection

from supervisely.project.project import (
    OpenMode,
    Dataset,
    read_single_project as read_project_wrapper,
)


from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
import supervisely.pointcloud.pointcloud as sly_pointcloud
from supervisely.project.video_project import VideoDataset, VideoProject
from supervisely.io.json import dump_json_file
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger


class PointcloudItemPaths(NamedTuple):
    #: :class:`str`: Full pointcloud file path of item
    pointcloud_path: str

    #: :class:`str`: Path to related images directory of item
    related_images_dir: str

    #: :class:`str`: Full annotation file path of item
    ann_path: str


class PointcloudItemInfo(NamedTuple):
    #: :class:`str`: Item's dataset name
    dataset_name: str

    #: :class:`str`: Item name
    name: str

    #: :class:`str`: Full pointcloud file path of item
    pointcloud_path: str

    #: :class:`str`: Path to related images directory of item
    related_images_dir: str

    #: :class:`str`: Full annotation file path of item
    ann_path: str


class PointcloudDataset(VideoDataset):
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

    annotation_class = PointcloudAnnotation
    item_info_type = PointcloudInfo

    @property
    def img_dir(self) -> str:
        """
        Not available for PointcloudDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Property 'img_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def img_info_dir(self):
        """
        Not available for PointcloudDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Property 'img_info_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def seg_dir(self):
        """
        Not available for PointcloudDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Property 'seg_dir' is not supported for {type(self).__name__} object."
        )

    def get_img_path(self, item_name: str) -> str:
        """
        Not available for PointcloudDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_img_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_img_info_path(self, img_name: str) -> str:
        """
        Not available for PointcloudDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_img_info_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_image_info(self, item_name: str) -> None:
        """
        Not available for PointcloudDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_image_info(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_seg_path(self, item_name: str) -> str:
        """
        Not available for PointcloudDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_seg_path(item_name)' is not supported for {type(self).__name__} object."
        )

    @property
    def pointcloud_dir(self) -> str:
        return self.item_dir

    @property
    def pointcloud_info_dir(self) -> str:
        return self.item_info_dir

    @staticmethod
    def _has_valid_ext(path: str) -> bool:
        return sly_pointcloud.has_valid_ext(path)

    def _get_empty_annotaion(self, item_name):
        return self.annotation_class()

    def get_pointcloud_path(self, item_name: str) -> str:
        """
        Path to the given pointcloud.

        :param item_name: Pointcloud name
        :type item_name: :class:`str`
        :return: Path to the given pointcloud
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = sly.PointcloudDataset(dataset_path, sly.OpenMode.READ)

            ds.get_pointcloud_path("PTC_0748")
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            ds.get_pointcloud_path("PTC_0748.pcd")
            # Output: '/home/admin/work/supervisely/projects/ptc_project/ds0/pointcloud/PTC_0748.pcd'
        """
        return super().get_item_path(item_name)

    def get_pointcloud_info(self, item_name: str) -> PointcloudInfo:
        """
        Information for Pointcloud with given name.

        :param item_name: Pointcloud name.
        :type item_name: str
        :return: Pointcloud with information for the given Dataset
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = sly.PointcloudDataset(dataset_path, sly.OpenMode.READ)

            info = ds.get_pointcloud_info("IMG_0748.pcd")
        """
        return self.get_item_info(item_name)

    def get_ann_path(self, item_name: str) -> str:
        """
        Path to the given annotation.

        :param item_name: PointcloudAnnotation name.
        :type item_name: str
        :return: Path to the given annotation
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = sly.PointcloudDataset(dataset_path, sly.OpenMode.READ)

            ds.get_ann_path("PTC_0748")
            # Output: RuntimeError: Item PTC_0748 not found in the project.

            ds.get_ann_path("PTC_0748.pcd")
            # Output: '/home/admin/work/supervisely/projects/ptc_project/ds0/ann/IMG_0748.pcd.json'
        """
        return super().get_ann_path(item_name)

    def delete_item(self, item_name: str) -> bool:
        """
        Delete pointcloud, annotation, pointcloud info and related images from PointcloudDataset.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: True if successful, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = sly.PointcloudDataset(dataset_path, sly.OpenMode.READ)

            result = dataset.delete_item("PTC_0748.pcd")
            # Output: True
        """
        if self.item_exists(item_name):
            data_path, rel_images_dir, ann_path = self.get_item_paths(item_name)
            img_info_path = self.get_pointcloud_info_path(item_name)
            silent_remove(data_path)
            silent_remove(ann_path)
            silent_remove(img_info_path)
            remove_dir(rel_images_dir)
            self._item_to_ann.pop(item_name)
            return True
        return False

    def add_item_file(
        self,
        item_name: str,
        item_path: str,
        ann: Optional[Union[PointcloudAnnotation, str]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        item_info: Optional[Union[PointcloudInfo, Dict, str]] = None,
    ) -> None:
        """
        Adds given item file to dataset items directory, and adds given annotation to dataset
        annotations directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param item_path: Path to the item.
        :type item_path: :class:`str`
        :param ann: PointcloudAnnotation object or path to annotation json file.
        :type ann: :class:`PointcloudAnnotation<supervisely.pointcloud_annotation.pointcloud_annotation.PointcloudAnnotation>` or :class:`str`, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: :class:`bool`, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: :class:`bool`, optional
        :param item_info: PointcloudInfo object or PointcloudInfo object converted to dict or path to item info json file for copying to dataset item info directory.
        :type item_info: :class:`PointcloudInfo<supervisely.api.pointcloud.pointcloud_api.PointcloudInfo>` or :class:`dict` or :class:`str`, optional
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if item_name already exists in dataset or item name has unsupported extension.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = sly.PointcloudDataset(dataset_path, sly.OpenMode.READ)

            ann = "/home/admin/work/supervisely/projects/ptc_project/ds0/ann/PTC_8888.pcd.json"
            ds.add_item_file("PTC_8888.pcd", "/home/admin/work/supervisely/projects/ptc_project/ds0/pointcloud/PTC_8888.pcd", ann=ann)
            print(ds.item_exists("PTC_8888.pcd"))
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

    def add_item_np(
        self,
        item_name: str,
        pointcloud: np.ndarray,
        ann: Optional[Union[PointcloudAnnotation, str]] = None,
        item_info: Optional[NamedTuple] = None,
    ) -> None:
        """
        Adds given numpy array as a pointcloud to dataset items directory, and adds given annotation to dataset ann directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: str
        :param pointcloud: numpy Pointcloud array [N, 3], in (X, Y, Z) format.
        :type pointcloud: np.ndarray
        :param ann: PointcloudAnnotation object or path to annotation .json file.
        :type ann: PointcloudAnnotation or str, optional
        :param item_info: NamedTuple PointcloudItemInfo containing information about Pointcloud.
        :type item_info: NamedTuple, optional
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`Exception` if item_name already exists in dataset or item name has unsupported extension
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = sly.PointcloudDataset(dataset_path, sly.OpenMode.READ)

            pointcloud_path = "/home/admin/Pointclouds/ptc0.pcd"
            img_np = sly.image.read(img_path)
            ds.add_item_np("IMG_050.jpeg", img_np)
        """
        # TODO: is it ok that names of params differs from base function?
        # TODO: check this function
        self._add_pointcloud_np(item_name, pointcloud)
        self._add_ann_by_type(item_name, ann)
        self._add_item_info(item_name, item_info)

    def add_item_raw_bytes(self, item_name, item_raw_bytes, ann=None, img_info=None):
        """
        Not available for PointcloudDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def _add_item_raw_bytes(self, item_name, item_raw_bytes):
        """
        Not available for PointcloudDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method '_add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def _add_pointcloud_np(self, item_name, pointcloud):
        if pointcloud is None:
            return
        self._check_add_item_name(item_name)
        dst_img_path = os.path.join(self.pointcloud_dir, item_name)
        sly_pointcloud.write(dst_img_path, pointcloud)

    def get_classes_stats(
        self,
        project_meta: Optional[ProjectMeta] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        if project_meta is None:
            project = PointcloudProject(self.project_dir, OpenMode.READ)
            project_meta = project.meta
        class_items = {}
        class_objects = {}
        class_figures = {}
        for obj_class in project_meta.obj_classes:
            class_items[obj_class.name] = 0
            class_objects[obj_class.name] = 0
            class_figures[obj_class.name] = 0
        objects_calculated = False
        for item_name in self:
            item_ann = self.get_ann(item_name, project_meta)
            item_class = {}
            if not objects_calculated:
                for ann_obj in item_ann.objects:
                    class_objects[ann_obj.obj_class.name] += 1
                objects_calculated = True
            for ptc_figure in item_ann.figures:
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

    def _validate_added_item_or_die(self, item_path):
        # Make sure we actually received a valid pointcloud file, clean it up and fail if not so.
        try:
            sly_pointcloud.validate_format(item_path)
        except (sly_pointcloud.UnsupportedPointcloudFormat, sly_pointcloud.PointcloudReadException):
            os.remove(item_path)
            raise

    def get_related_images_path(self, item_name: str) -> str:
        item_name_temp = item_name.replace(".", "_")
        rimg_dir = os.path.join(self.directory, self.related_images_dir_name, item_name_temp)
        return rimg_dir

    def get_item_paths(self, item_name: str) -> PointcloudItemPaths:
        return PointcloudItemPaths(
            pointcloud_path=self.get_pointcloud_path(item_name),
            related_images_dir=self.get_related_images_path(item_name),
            ann_path=self.get_ann_path(item_name),
        )

    # def validate_figure_bounds(self,)

    def get_ann(
        self, item_name, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudAnnotation:
        """
        Read pointcloud annotation of item from json.

        :param item_name: Pointcloud name.
        :type item_name: str
        :param project_meta: Project Meta.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudAnnotation object
        :rtype: :class:`PointcloudAnnotation<supervisely.PointcloudAnnotation>`
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
            # Output: PointcloudAnnotation
        """
        ann_path = self.get_ann_path(item_name)
        return PointcloudAnnotation.load_json_file(ann_path, project_meta, key_id_map)

    def get_related_images(self, item_name: str) -> List[Tuple[str, Dict]]:
        results = []
        path = self.get_related_images_path(item_name)
        if dir_exists(path):
            files = list_files(path, sly_image.SUPPORTED_IMG_EXTS)
            for file in files:
                img_meta_path = os.path.join(path, get_file_name_with_ext(file) + ".json")
                img_meta = {}
                if file_exists(img_meta_path):
                    img_meta = load_json_file(img_meta_path)
                    if img_meta[ApiField.NAME] != get_file_name_with_ext(file):
                        raise RuntimeError("Wrong format: name field contains wrong image path")
                results.append((file, img_meta))
        return results

    def get_pointcloud_info_path(self, item_name: str) -> str:
        return self.get_item_info_path(item_name)


class PointcloudProject(VideoProject):
    """
    PointcloudProject is a parent directory for pointcloud datasets. PointcloudProject object is immutable.

    :param directory: Path to pointcloud project directory.
    :type directory: :class:`str`
    :param mode: Determines working mode for the given project.
    :type mode: :class:`OpenMode<supervisely.project.project.OpenMode>`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        project_path = "/home/admin/work/supervisely/projects/ptc_project"
        project = sly.PointcloudProject(project_path, sly.OpenMode.READ)
    """

    dataset_class = PointcloudDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = PointcloudDataset

    @classmethod
    def read_single(cls, dir) -> PointcloudProject:
        return read_project_wrapper(dir, cls)

    def get_classes_stats(
        self,
        dataset_names: Optional[List[str]] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        return super(PointcloudProject, self).get_classes_stats(
            dataset_names, return_objects_count, return_figures_count, return_items_count
        )

    @staticmethod
    def get_train_val_splits_by_count(
        project_dir: str, train_count: int, val_count: int
    ) -> Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]:
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
        :rtype: :class:`Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]`
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudProject
            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = PointcloudProject(project_path, sly.OpenMode.READ)
            train_count = 4
            val_count = 2
            train_items, val_items = project.get_train_val_splits_by_count(project_path, train_count, val_count)
        """

        def _list_items_for_splits(project) -> List[PointcloudItemInfo]:
            items = []
            for dataset in project.datasets:
                for item_name in dataset:
                    items.append(
                        PointcloudItemInfo(
                            dataset_name=dataset.name,
                            name=item_name,
                            pointcloud_path=dataset.get_pointcloud_path(item_name),
                            related_images_dir=dataset.get_related_images_path(item_name),
                            ann_path=dataset.get_ann_path(item_name),
                        )
                    )
            return items

        project = PointcloudProject(project_dir, OpenMode.READ)
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
    ) -> Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]:
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
        :rtype: :class:`Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = sly.PointcloudProject(project_path, sly.OpenMode.READ)
            train_tag_name = 'train'
            val_tag_name = 'val'
            train_items, val_items = project.get_train_val_splits_by_tag(project_path, train_tag_name, val_tag_name)
        """
        untagged_actions = ["ignore", "train", "val"]
        if untagged not in untagged_actions:
            raise ValueError(
                f"Unknown untagged action {untagged}. Should be one of {untagged_actions}"
            )
        project = PointcloudProject(project_dir, OpenMode.READ)
        train_items = []
        val_items = []
        for dataset in project.datasets:
            for item_name in dataset:
                item_paths = dataset.get_item_paths(item_name)
                info = PointcloudItemInfo(
                    dataset_name=dataset.name,
                    name=item_name,
                    pointcloud_path=item_paths.pointcloud_path,
                    related_images_dir=item_paths.related_images_dir,
                    ann_path=item_paths.ann_path,
                )

                ann = PointcloudAnnotation.load_json_file(item_paths.ann_path, project.meta)
                if ann.tags.get(train_tag_name) is not None:
                    train_items.append(info)
                if ann.tags.get(val_tag_name) is not None:
                    val_items.append(info)
                if ann.tags.get(train_tag_name) is None and ann.tags.get(val_tag_name) is None:
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
    ) -> Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]:
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
        :rtype: :class:`Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = sly.PointcloudProject(project_path, sly.OpenMode.READ)
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
                    info = PointcloudItemInfo(
                        dataset_name=dataset.name,
                        name=item_name,
                        pointcloud_path=item_paths.pointcloud_path,
                        related_images_dir=item_paths.related_images_dir,
                        ann_path=item_paths.ann_path,
                    )
                    items_list.append(info)

        project = PointcloudProject(project_dir, OpenMode.READ)
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
    ) -> PointcloudProject:
        """
        Download pointcloud project from Supervisely to the given directory.

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

                # Local destination Pointcloud Project folder
                save_directory = "/home/admin/work/supervisely/source/ptc_project"

                # Obtain server address and your api_token from environment variables
                # Edit those values if you run this notebook on your own PC
                address = os.environ['SERVER_ADDRESS']
                token = os.environ['API_TOKEN']

                # Initialize API object
                api = sly.Api(address, token)
                project_id = 8888

                # Download Project
                sly.PointcloudProject.download(api, project_id, save_directory)
                project_fs = sly.PointcloudProject(save_directory, sly.OpenMode.READ)
        """
        download_pointcloud_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            download_items=download_pointclouds,
            download_related_images=download_related_images,
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
        Uploads pointcloud project to Supervisely from the given directory.

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
            project_directory = "/home/admin/work/supervisely/source/ptc_project"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)

            # Upload Pointcloud Project
            project_id, project_name = sly.PointcloudProject.upload(
                project_directory,
                api,
                workspace_id=45,
                project_name="My Pointcloud Project"
            )
        """
        return upload_pointcloud_project(
            directory=directory,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )


def download_pointcloud_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_items: Optional[bool] = True,
    download_related_images: Optional[bool] = True,
    download_pointclouds_info: Optional[bool] = False,
    batch_size: Optional[int] = 10,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """
    Download pointcloud project to the local directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID to download.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param download_items: Include pointcloud items in the download.
    :type download_items: bool, optional
    :param download_related_images: Include related context images of a pointcloud project in the download.
    :type download_related_images: bool, optional
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

        # Download pointcloud project
        project_id = 19542
        project_info = api.project.get_info_by_id(project_id)
        num_pointclouds = project_info.items_count

        p = tqdm(
            desc="Downloading pointcloud project",
            total=num_pointclouds,
        )
        sly.download_pointcloud_project(
            api,
            project_id,
            dest_dir,
            progress_cb=p,
        )
    """
    key_id_map = KeyIdMap()

    project_fs = PointcloudProject(dest_dir, OpenMode.CREATE)

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
        pointclouds = api.pointcloud.get_list(dataset.id)

        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset.name), total_cnt=len(pointclouds)
            )
        for batch in batched(pointclouds, batch_size=batch_size):
            pointcloud_ids = [pointcloud_info.id for pointcloud_info in batch]
            pointcloud_names = [pointcloud_info.name for pointcloud_info in batch]

            ann_jsons = api.pointcloud.annotation.download_bulk(dataset.id, pointcloud_ids)

            for pointcloud_id, pointcloud_name, ann_json, pointcloud_info in zip(
                pointcloud_ids, pointcloud_names, ann_jsons, batch
            ):
                if pointcloud_name != ann_json[ApiField.NAME]:
                    raise RuntimeError("Error in api.video.annotation.download_batch: broken order")

                pointcloud_file_path = dataset_fs.generate_item_path(pointcloud_name)
                if download_items:
                    try:
                        api.pointcloud.download_path(pointcloud_id, pointcloud_file_path)
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
                        related_images = api.pointcloud.get_list_related_images(pointcloud_id)
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
                            api.pointcloud.download_related_image(rimage_id, path_img)
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

                pointcloud_file_path = pointcloud_file_path if download_items else None
                pointcloud_info = pointcloud_info._asdict() if download_pointclouds_info else None
                try:
                    pointcloud_ann = PointcloudAnnotation.from_json(
                        ann_json, project_fs.meta, key_id_map
                    )
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "project_id": project_id,
                            "dataset_id": dataset.id,
                            "pointcloud_id": pointcloud_id,
                            "pointcloud_name": pointcloud_name,
                            "ann_json": ann_json,
                        },
                    )
                    raise e
                try:
                    dataset_fs.add_item_file(
                        pointcloud_name,
                        pointcloud_file_path,
                        ann=pointcloud_ann,
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


def upload_pointcloud_project(
    directory: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> Tuple[int, str]:
    project_fs = PointcloudProject.read_single(directory)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.POINT_CLOUDS)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    key_id_map = KeyIdMap()
    for dataset_fs in project_fs:
        dataset = api.dataset.create(project.id, dataset_fs.name, change_name_if_conflict=True)

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Uploading dataset: {!r}".format(dataset.name), total_cnt=len(dataset_fs)
            )

        for item_name in dataset_fs:
            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)
            related_items = dataset_fs.get_related_images(item_name)

            try:
                _, meta = related_items[0]
                timestamp = meta[ApiField.META]["timestamp"]
                if timestamp:
                    item_meta = {"timestamp": timestamp}
            except (KeyError, IndexError):
                item_meta = {}

            try:
                pointcloud = api.pointcloud.upload_path(dataset.id, item_name, item_path, item_meta)
            except Exception as e:
                logger.info(
                    "INFO FOR DEBUGGING",
                    extra={
                        "project_id": project.id,
                        "dataset_id": dataset.id,
                        "item_name": item_name,
                        "item_path": item_path,
                        "item_meta": item_meta,
                    },
                )
                raise e

            # validate_item_annotation
            ann_json = load_json_file(ann_path)
            try:
                ann = PointcloudAnnotation.from_json(ann_json, project_fs.meta)
            except Exception as e:
                logger.info(
                    "INFO FOR DEBUGGING",
                    extra={
                        "project_id": project.id,
                        "dataset_id": dataset.id,
                        "pointcloud_id": pointcloud.id,
                        "pointcloud_name": pointcloud.name,
                        "ann_json": ann_json,
                    },
                )
                raise e

            try:
                # ignore existing key_id_map because the new objects will be created
                api.pointcloud.annotation.append(pointcloud.id, ann, key_id_map)
            except Exception as e:
                logger.info(
                    "INFO FOR DEBUGGING",
                    extra={
                        "project_id": project.id,
                        "dataset_id": dataset.id,
                        "pointcloud_id": pointcloud.id,
                        "pointcloud_name": pointcloud.name,
                        "ann": ann.to_json(),
                    },
                )
                raise e

            # upload related_images if exist
            if len(related_items) != 0:
                rimg_infos = []
                camera_names = []
                for img_ind, (img_path, meta_json) in enumerate(related_items):
                    try:
                        img = api.pointcloud.upload_related_image(img_path)
                    except Exception as e:
                        logger.info(
                            "INFO FOR DEBUGGING",
                            extra={
                                "project_id": project.id,
                                "dataset_id": dataset.id,
                                "pointcloud_id": pointcloud.id,
                                "pointcloud_name": pointcloud.name,
                                "img_path": img_path,
                            },
                        )
                        raise e
                    if "deviceId" not in meta_json[ApiField.META].keys():
                        camera_names.append(f"CAM_{str(img_ind).zfill(2)}")
                    else:
                        camera_names.append(meta_json[ApiField.META]["deviceId"])
                    rimg_infos.append(
                        {
                            ApiField.ENTITY_ID: pointcloud.id,
                            ApiField.NAME: meta_json[ApiField.NAME],
                            ApiField.HASH: img,
                            ApiField.META: meta_json[ApiField.META],
                        }
                    )

                try:
                    api.pointcloud.add_related_images(rimg_infos, camera_names)
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "project_id": project.id,
                            "dataset_id": dataset.id,
                            "pointcloud_id": pointcloud.id,
                            "pointcloud_name": pointcloud.name,
                            "rimg_infos": rimg_infos,
                            "camera_names": camera_names,
                        },
                    )
                    raise e
            if log_progress:
                ds_progress.iters_done_report(1)
            if progress_cb is not None:
                progress_cb(1)

    return project.id, project_name
