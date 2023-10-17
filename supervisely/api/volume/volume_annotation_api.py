# coding: utf-8

import json
from typing import List, Optional, Union, Callable

from tqdm import tqdm


from supervisely.project.project_meta import ProjectMeta
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely.volume_annotation.volume_object import VolumeObject
from supervisely.geometry.mask_3d import Mask3D
from supervisely.geometry.any_geometry import AnyGeometry

from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.io.json import load_json_file


class VolumeAnnotationAPI(EntityAnnotationAPI):
    """
    :class:`VolumeAnnotation<supervisely.volume_annotation.volume_annotation.VolumeAnnotation>` for a single volume. :class:`VolumeAnnotationAPI<VolumeAnnotationAPI>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import supervisely as sly

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

        volume_id = 19581134
        ann_info = api.volume.annotation.download(volume_id)
    """

    _method_download_bulk = "volumes.annotations.bulk.info"
    _entity_ids_str = ApiField.VOLUME_IDS

    def download(self, volume_id: int):
        """
        Download information about VolumeAnnotation by volume ID from API.
        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :return: Information about VolumeAnnotation in json format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from pprint import pprint

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19581134
            ann_info = api.volume.annotation.download(volume_id)
            print(ann_info)
            # Output:
            # {
            #     'createdAt': '2023-03-29T12:30:37.078Z',
            #     'datasetId': 61803,
            #     'description': '',
            #     'objects': [],
            #     'planes': [],
            #     'spatialFigures': [],
            #     'tags': [{'createdAt': '2023-04-03T13:21:53.368Z',
            #             'id': 12259702,
            #             'labelerLogin': 'almaz',
            #             'name': 'info',
            #             'tagId': 385328,
            #             'updatedAt': '2023-04-03T13:21:53.368Z',
            #             'value': 'age 31'}],
            #     'updatedAt': '2023-03-29T12:30:37.078Z',
            #     'volumeId': 19581134,
            #     'volumeMeta': {
            #             'ACS': 'RAS',
            #             'IJK2WorldMatrix': [0.7617, 0, 0,
            #                                 -194.2384, 0, 0.76171,
            #                                 0, -217.5384, 0,
            #                                 0, 2.5, -347.75,
            #                                 0, 0, 0, 1],
            #             'channelsCount': 1,
            #             'dimensionsIJK': {'x': 512, 'y': 512, 'z': 139},
            #             'intensity': {'max': 3071, 'min': -3024},
            #             'rescaleIntercept': 0,
            #             'rescaleSlope': 1,
            #             'windowCenter': 23.5,
            #             'windowWidth': 6095
            # },
            #     'volumeName': 'CTChest.nrrd'
            # }
        """

        volume_info = self._api.volume.get_info_by_id(volume_id)
        return self._download(volume_info.dataset_id, volume_id)

    def append(self, volume_id: int, ann: VolumeAnnotation, key_id_map: KeyIdMap = None):
        """
        Loads VolumeAnnotation to a given volume ID in the API.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :param ann: VolumeAnnotation object.
        :type ann: VolumeAnnotation
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19581134
            api.volume.annotation.append(volume_id, volume_ann)
        """
        if ann.spatial_figures:
            figures = ann.figures + ann.spatial_figures
        else:
            figures = ann.figures

        info = self._api.volume.get_info_by_id(volume_id)
        self._append(
            self._api.volume.tag,
            self._api.volume.object,
            self._api.volume.figure,
            info.project_id,
            info.dataset_id,
            volume_id,
            ann.tags,
            ann.objects,
            figures,
            key_id_map,
        )

    def upload_paths(
        self,
        volume_ids: List[int],
        ann_paths: List[str],
        project_meta: ProjectMeta,
        interpolation_dirs=None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ):
        """
        Loads VolumeAnnotations from a given paths to a given volumes IDs in the API. Volumes IDs must be from one dataset.

        :param volume_ids: Volumes IDs in Supervisely.
        :type volume_ids: List[int]
        :param ann_paths: Paths to annotations on local machine.
        :type ann_paths: List[str]
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>` for VolumeAnnotations.
        :type project_meta: ProjectMeta
        :param interpolation_dirs:
        :type interpolation_dirs:
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

            volume_ids = [121236918, 121236919]
            ann_pathes = ['/home/admin/work/supervisely/example/ann1.json', '/home/admin/work/supervisely/example/ann2.json']
            api.volume.annotation.upload_paths(volume_ids, ann_pathes, meta)
        """

        if interpolation_dirs is None:
            interpolation_dirs = [None] * len(ann_paths)

        key_id_map = KeyIdMap()
        for volume_id, ann_path, interpolation_dir in zip(
            volume_ids, ann_paths, interpolation_dirs
        ):
            ann_json = load_json_file(ann_path)
            ann = VolumeAnnotation.from_json(ann_json, project_meta)
            self.append(volume_id, ann, key_id_map)

            # upload existing interpolations or create on the fly and and add them to empty mesh figures
            self._api.volume.figure.upload_stl_meshes(
                volume_id, ann.spatial_figures, key_id_map, interpolation_dir
            )
            if progress_cb is not None:
                progress_cb(1)

    def append_objects(
        self,
        volume_id: int,
        objects: Union[List[VolumeObject], VolumeObjectCollection],
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        """
        Add new VolumeObjects to a volume annotation in Supervisely project.

        :param volume_id: The ID of the volume.
        :type volume_id: int
        :param objects: New volume objects.
        :type objects: List[VolumeObject] or VolumeObjectCollection
        :param key_id_map: The KeyIdMap (optional).
        :type key_id_map: KeyIdMap, optional
        :return: None
        :rtype: NoneType

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

            volume_id = 151344
            volume_info = api.volume.get_info_by_id(volume_id)
            mask_3d_path = "data/mask/lung.nrrd"
            lung_obj_class = sly.ObjClass("lung", sly.Mask3D)
            lung = sly.VolumeObject(lung_obj_class, mask_3d=mask_3d_path)
            objects = sly.VolumeObjectCollection([lung])
            api.volume.annotation.append_objects(volume_info.id, objects)
        """

        sf_figures = []
        for volume_object in objects:
            if volume_object.obj_class.geometry_type in (Mask3D, AnyGeometry):
                if isinstance(volume_object.figure.geometry, Mask3D):
                    sf_figures.append(volume_object.figure)

        volume_meta = self._api.volume.get_info_by_id(volume_id).meta
        ann = VolumeAnnotation(volume_meta, objects, spatial_figures=sf_figures)
        self.append(volume_id, ann, key_id_map)
