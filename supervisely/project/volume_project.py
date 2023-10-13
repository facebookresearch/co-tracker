# coding: utf-8

from collections import namedtuple
from typing import Optional, List, Callable, Tuple, Union
import os
import nrrd

from tqdm import tqdm

from supervisely.io.fs import file_exists, touch, remove_dir
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely._utils import batched
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.volume import volume as sly_volume

from supervisely.project.project import Dataset, Project, OpenMode
from supervisely.project.video_project import VideoDataset, VideoProject
from supervisely.project.project import read_single_project as read_project_wrapper
from supervisely.project.project_type import ProjectType
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.geometry.mask_3d import Mask3D

VolumeItemPaths = namedtuple("VolumeItemPaths", ["volume_path", "ann_path"])


class VolumeDataset(VideoDataset):
    item_dir_name = "volume"
    interpolation_dir = "interpolation"
    annotation_class = VolumeAnnotation
    item_module = sly_volume
    paths_tuple = VolumeItemPaths

    @classmethod
    def _has_valid_ext(cls, path: str) -> bool:
        """
        Checks if file from given path is supported
        :param path: str
        :return: bool
        """
        return sly_volume.has_valid_ext(path)

    def _get_empty_annotaion(self, item_name):
        path = item_name
        _, volume_meta = sly_volume.read_nrrd_serie_volume(path)
        return self.annotation_class(volume_meta)

    def get_interpolation_dir(self, item_name):
        return os.path.join(self.directory, self.interpolation_dir, item_name)

    def get_interpolation_path(self, item_name, figure):
        return os.path.join(self.get_interpolation_dir(item_name), figure.key().hex + ".stl")

    def get_classes_stats(
        self,
        project_meta: Optional[ProjectMeta] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        if project_meta is None:
            project = VolumeProject(self.project_dir, OpenMode.READ)
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
            for volume_figure in item_ann.figures:
                class_figures[volume_figure.parent_object.obj_class.name] += 1
                item_class[volume_figure.parent_object.obj_class.name] = True
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


class VolumeProject(VideoProject):
    dataset_class = VolumeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = VolumeDataset

    def get_classes_stats(
        self,
        dataset_names: Optional[List[str]] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        return super(VolumeProject, self).get_classes_stats(
            dataset_names, return_objects_count, return_figures_count, return_items_count
        )

    @staticmethod
    def download(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: Optional[List[int]] = None,
        download_volumes: Optional[bool] = True,
        log_progress: Optional[bool] = False,
    ) -> None:
        """
        Download volume project from Supervisely to the given directory.

        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Supervisely downloadable project ID.
        :type project_id: :class:`int`
        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`
        :param dataset_ids: Dataset IDs.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param download_volumes: Download volume data files or not.
        :type download_volumes: :class:`bool`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

        .. code-block:: python

                import supervisely as sly

                # Local destination Volume Project folder
                save_directory = "/home/admin/work/supervisely/source/vlm_project"

                # Obtain server address and your api_token from environment variables
                # Edit those values if you run this notebook on your own PC
                address = os.environ['SERVER_ADDRESS']
                token = os.environ['API_TOKEN']

                # Initialize API object
                api = sly.Api(address, token)
                project_id = 8888

                # Download Project
                sly.VolumeProject.download(api, project_id, save_directory)
                project_fs = sly.VolumeProject(save_directory, sly.OpenMode.READ)
        """
        download_volume_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            download_volumes=download_volumes,
            log_progress=log_progress,
        )

    @staticmethod
    def upload(
        directory: str,
        api: Api,
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: Optional[bool] = False,
    ) -> Tuple[int, str]:
        """
        Uploads volume project to Supervisely from the given directory.

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
        :return: Project ID and name. It is recommended to check that returned project name coincides with provided project name.
        :rtype: :class:`int`, :class:`str`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local folder with Volume Project
            project_directory = "/home/admin/work/supervisely/source/vlm_project"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)

            # Upload Volume Project
            project_id, project_name = sly.VolumeProject.upload(
                project_directory,
                api,
                workspace_id=45,
                project_name="My Volume Project"
            )
        """
        return upload_volume_project(
            dir=directory,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
        )


def download_volume_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_volumes: Optional[bool] = True,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """
    Download volume project to the local directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID to download.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param download_volumes: Include volumes in the download.
    :type download_volumes: bool, optional
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

        # Download volume project
        project_id = 18532
        project_info = api.project.get_info_by_id(project_id)
        num_volumes = project_info.items_count

        p = tqdm(desc="Downloading volume project", total=num_volumes)
        sly.download_volume_project(
            api,
            project_id,
            dest_dir,
            progress_cb=p,
        )
    """

    LOG_BATCH_SIZE = 1

    key_id_map = KeyIdMap()

    project_fs = VolumeProject(dest_dir, OpenMode.CREATE)

    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    datasets_infos = []
    if dataset_ids is not None:
        for ds_id in dataset_ids:
            datasets_infos.append(api.dataset.get_info_by_id(ds_id))
    else:
        datasets_infos = api.dataset.get_list(project_id)

    for dataset in datasets_infos:
        dataset_fs: VolumeDataset = project_fs.create_dataset(dataset.name)
        volumes = api.volume.get_list(dataset.id)

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset.name), total_cnt=len(volumes)
            )
        for batch in batched(volumes, batch_size=LOG_BATCH_SIZE):
            volume_ids = [volume_info.id for volume_info in batch]
            volume_names = [volume_info.name for volume_info in batch]

            ann_jsons = api.volume.annotation.download_bulk(dataset.id, volume_ids)

            for volume_id, volume_name, volume_info, ann_json in zip(
                volume_ids, volume_names, batch, ann_jsons
            ):
                if volume_name != ann_json[ApiField.VOLUME_NAME]:
                    raise RuntimeError(
                        "Error in api.volume.annotation.download_batch: broken order"
                    )

                volume_file_path = dataset_fs.generate_item_path(volume_name)
                if download_volumes is True:
                    item_progress = None
                    if log_progress:
                        item_progress = Progress(
                            f"Downloading {volume_name}",
                            total_cnt=volume_info.sizeb,
                            is_size=True,
                        )
                        api.volume.download_path(
                            volume_id, volume_file_path, item_progress.iters_done_report
                        )
                    else:
                        api.volume.download_path(volume_id, volume_file_path)
                else:
                    touch(volume_file_path)

                ann = VolumeAnnotation.from_json(ann_json, project_fs.meta, key_id_map)

                for sf in ann.spatial_figures:
                    if sf.geometry.name() == Mask3D.name():
                        load_figure_data(api, volume_file_path, sf, key_id_map)

                dataset_fs.add_item_file(
                    volume_name,
                    volume_file_path,
                    ann=ann,
                    _validate_item=False,
                )

                mesh_ids = []
                mesh_paths = []
                for sf in ann.spatial_figures:
                    figure_id = key_id_map.get_figure_id(sf.key())
                    mesh_ids.append(figure_id)
                    figure_path = dataset_fs.get_interpolation_path(volume_name, sf)
                    mesh_paths.append(figure_path)
                api.volume.figure.download_stl_meshes(mesh_ids, mesh_paths)
            if log_progress:
                ds_progress.iters_done_report(len(batch))
            if progress_cb is not None:
                progress_cb(len(batch))
    project_fs.set_key_id_map(key_id_map)


def load_figure_data(
    api: Api, volume_file_path: str, spatial_figure: VolumeFigure, key_id_map: KeyIdMap
):
    """
    Load data into figure geometry.

    :param api: Supervisely API address and token.
    :type api: Api
    :param volume_file_path: Path to Volume file location
    :type volume_file_path: str
    :param spatial_figure: Spatial figure
    :type spatial_figure: VolumeFigure object
    :param key_id_map: Mapped keys and IDs
    :type key_id_map: KeyIdMap object
    """
    figure_id = key_id_map.get_figure_id(spatial_figure.key())
    figure_path = "{}_mask3d/".format(volume_file_path[:-5]) + f"{figure_id}.nrrd"
    api.volume.figure.download_stl_meshes([figure_id], [figure_path])
    Mask3D.from_file(spatial_figure, figure_path)


# TODO: add methods to convert to 3d masks


def upload_volume_project(dir, api: Api, workspace_id, project_name=None, log_progress=True):
    project_fs = VolumeProject.read_single(dir)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.VOLUMES)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    for dataset_fs in project_fs.datasets:
        dataset_fs: VolumeDataset
        dataset = api.dataset.create(project.id, dataset_fs.name)

        names, item_paths, ann_paths, interpolation_dirs = [], [], [], []
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            item_paths.append(img_path)
            ann_paths.append(ann_path)
            interpolation_dirs.append(dataset_fs.get_interpolation_dir(item_name))

        progress_cb = None
        if log_progress:
            ds_progress = Progress(
                "Uploading volumes to dataset {!r}".format(dataset.name),
                total_cnt=len(item_paths),
            )
            progress_cb = ds_progress.iters_done_report

        item_infos = api.volume.upload_nrrd_series_paths(
            dataset.id, names, item_paths, progress_cb, log_progress
        )
        item_ids = [item_info.id for item_info in item_infos]
        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Uploading annotations to dataset {!r}".format(dataset.name),
                total_cnt=len(item_paths),
            )
            progress_cb = ds_progress.iters_done_report

        api.volume.annotation.upload_paths(
            item_ids, ann_paths, project_fs.meta, interpolation_dirs, progress_cb
        )

    return project.id, project.name
