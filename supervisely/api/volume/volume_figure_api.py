# coding: utf-8
import re
import os
from typing import List, Dict
from requests_toolbelt import MultipartDecoder, MultipartEncoder
from supervisely.io.fs import ensure_base_path, file_exists
from supervisely._utils import batched
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.volume_annotation.plane import Plane
from supervisely.geometry.mask_3d import Mask3D
import supervisely.volume_annotation.constants as constants
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh


class VolumeFigureApi(FigureApi):
    """
    :class:`VolumeFigure<supervisely.volume_annotation.volume_figure.VolumeFigure>` for a single volume.
    """

    def create(
        self,
        volume_id: int,
        object_id: int,
        plane_name: str,
        slice_index: int,
        geometry_json: dict,
        geometry_type,
        # track_id=None,
    ):
        """
        Create new VolumeFigure for given slice in given volume ID.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :param object_id: ID of the object to which the VolumeFigure belongs.
        :type object_id: int
        :param plane_name: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>` of the slice in volume.
        :type plane_name: str
        :param slice_index: Number of the slice to add VolumeFigure.
        :type slice_index: int
        :param geometry_json: Parameters of geometry for VolumeFigure.
        :type geometry_json: dict
        :param geometry_type: Type of VolumeFigure geometry.
        :type geometry_type: str
        :return: New figure ID
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.volume_annotation.plane import Plane

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19581134
            object_id = 5565016
            slice_index = 0
            plane_name = Plane.AXIAL
            geometry_json = {'points': {'exterior': [[500, 500], [1555, 1500]], 'interior': []}}
            geometry_type = 'rectangle'

            figure_id = api.volume.figure.create(
                volume_id,
                object_id,
                plane_name,
                slice_index,
                geometry_json,
                geometry_type
            ) # 87821207
        """

        Plane.validate_name(plane_name)

        return super().create(
            volume_id,
            object_id,
            {
                constants.SLICE_INDEX: slice_index,
                constants.NORMAL: Plane.get_normal(plane_name),
                # for backward compatibility
                ApiField.META: {
                    constants.SLICE_INDEX: slice_index,
                    constants.NORMAL: Plane.get_normal(plane_name),
                },
            },
            geometry_json,
            geometry_type,
            # track_id,
        )

    def append_bulk(self, volume_id: int, figures: List[VolumeFigure], key_id_map: KeyIdMap):
        """
        Add VolumeFigures to given Volume by ID.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param figures: List of VolumeFigure objects.
        :type figures: List[VolumeFigure]
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.volume_annotation.plane import Plane

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 19370
            volume_id = 19617444

            key_id_map = sly.KeyIdMap()

            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            vol_ann_json = api.volume.annotation.download(volume_id)
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)
            volume_obj_collection = vol_ann.objects.to_json()
            vol_obj = sly.VolumeObject.from_json(volume_obj_collection[1], project_meta)

            figure = sly.VolumeFigure(
                vol_obj,
                sly.Rectangle(20, 20, 129, 200),
                sly.Plane.AXIAL,
                45,
            )

            api.volume.figure.append_bulk(volume_id, [figure], key_id_map)
        """

        if len(figures) == 0:
            return
        keys = []
        keys_mask3d = []
        figures_json = []
        figures_mask3d_json = []
        for figure in figures:
            if figure.geometry.name() == Mask3D.name():
                keys_mask3d.append(figure.key())
                figures_mask3d_json.append(figure.to_json(key_id_map, save_meta=True))
            else:
                keys.append(figure.key())
                figures_json.append(figure.to_json(key_id_map, save_meta=True))
        # Figure is missing required field \"meta.normal\"","index":0}}
        self._append_bulk(volume_id, figures_json, keys, key_id_map)
        if len(figures_mask3d_json) != 0:
            self._append_bulk_mask3d(volume_id, figures_mask3d_json, keys_mask3d, key_id_map)

    def _download_geometries_batch(self, ids: List[int]):
        """
        Private method. Download figures geometries with given IDs from storage.
        """

        for batch_ids in batched(ids):
            response = self._api.post("figures.bulk.download.geometry", {ApiField.IDS: batch_ids})
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                figure_id = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])
                yield figure_id, part

    def download_stl_meshes(self, ids: List[int], paths: List[str]):
        """
        Download STL meshes for the specified figure IDs and saves them to the specified paths.

        :param ids: VolumeFigure ID in Supervisely.
        :type ids: int
        :param paths: List of paths to download.
        :type paths: List[str]
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            STORAGE_DIR = sly.app.get_data_dir()

            volume_id = 19371414
            project_id = 17215

            volume = api.volume.get_info_by_id(volume_id)

            key_id_map = sly.KeyIdMap()
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            vol_ann_json = api.volume.annotation.download(volume_id)
            id_to_paths = {}
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)

            for sp_figure in vol_ann.spatial_figures:
                figure_id = key_id_map.get_figure_id(sp_figure.key())
                id_to_paths[figure_id] = f"{STORAGE_DIR}/{figure_id}.stl"
            if id_to_paths:
                api.volume.figure.download_stl_meshes(*zip(*id_to_paths.items()))
        """

        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise RuntimeError('Can not match "ids" and "paths" lists, len(ids) != len(paths)')

        id_to_path = {id: path for id, path in zip(ids, paths)}
        for img_id, resp_part in self._download_geometries_batch(ids):
            ensure_base_path(id_to_path[img_id])
            with open(id_to_path[img_id], "wb") as w:
                w.write(resp_part.content)

    def interpolate(self, volume_id: int, spatial_figure: VolumeFigure, key_id_map: KeyIdMap):
        """
        Interpolate a spatial figure with a ClosedSurfaceMesh geometry.

        :param volume_id: VolumeFigure ID in Supervisely.
        :type volume_id: int
        :param spatial_figure: Spatial figure to interpolate.
        :type spatial_figure: VolumeFigure
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19371414
            project_id = 17215

            volume = api.volume.get_info_by_id(volume_id)

            key_id_map = sly.KeyIdMap()
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            vol_ann_json = api.volume.annotation.download(volume_id)
            id_to_paths = {}
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)

            for sp_figure in vol_ann.spatial_figures:
                res = volume_figure_api.interpolate(volume_id, sp_figure, key_id_map)
        """

        if type(spatial_figure._geometry) != ClosedSurfaceMesh:
            raise TypeError(
                "Interpolation can be created only for figures with geometry ClosedSurfaceMesh"
            )
        object_id = key_id_map.get_object_id(spatial_figure.volume_object.key())
        response = self._api.post(
            "figures.volumetric_interpolation",
            {ApiField.VOLUME_ID: volume_id, ApiField.OBJECT_ID: object_id},
        )
        return response.content

    # def interpolate_batch(
    #     self, volume_id, spatial_figures: List[VolumeFigure], key_id_map: KeyIdMap
    # ):
    #     # raise NotImplementedError()
    #     # STL mesh interpolations can not be uploaded:
    #     # 400 Client Error: Bad Request for url: public/api/v3/figures.bulk.add ({"error":"Please, use \"figures.bulk.upload.geometry\" method to update figures with \"geometryType\" closed_surface_mesh","details":{"figures":[14]}})
    #     # figures.bulk.upload.geometry

    #     meshes = []
    #     for mesh in spatial_figures:
    #         object_id = key_id_map.get_object_id(mesh.volume_object.key())
    #         response = self._api.post(
    #             "figures.volumetric_interpolation",
    #             {ApiField.VOLUME_ID: volume_id, ApiField.OBJECT_ID: object_id},
    #         )
    #         figure_id = key_id_map.get_figure_id(mesh.key())

    #         # @TODO: load from disk or get from server
    #         meshes.append(response.json())

    #     # for batch in batched(items_to_upload):
    #     #     content_dict = {}
    #     #     for idx, item in enumerate(batch):
    #     #         content_dict["{}-file".format(idx)] = (str(idx), func_item_to_byte_stream(item), 'nrrd/*')
    #     #     encoder = MultipartEncoder(fields=content_dict)
    #     #     self._api.post('import-storage.bulk.upload', encoder)
    #     #     if progress_cb is not None:
    #     #         progress_cb(len(batch))

    #     # content_dict = {}
    #     # for idx, item in enumerate(meshes):
    #     #     content_dict[f"{idx}-mesh"] = (
    #     #         str(idx),
    #     #         func_item_to_byte_stream(item),
    #     #     )
    #     # encoder = MultipartEncoder(fields=content_dict)

    #     # self._api.post("figures.bulk.upload.geometry", encoder)

    #     # # if progress_cb is not None:
    #     # #     progress_cb(len(batch))

    #     # #     self._api.post(
    #     # #         "figures.bulk.upload.geometry",
    #     # #         {ApiField.FIGURE_ID: figure_id, ApiField.GEOMETRY: response.json()},
    #     # #     )

    #     # #     results.append(response.json())
    #     # return results

    # def _upload_geometries_batch(ids, )
    def _upload_meshes_batch(self, figure2bytes):
        """
        Private method. Upload figures geometry by given ID to storage.

        :param figure2bytes: Dictionary with figures IDs and geometries.
        :type figure2bytes: dict
        :rtype: :class:`NoneType`
        :Usage example:
        """

        for figure_id, figure_bytes in figure2bytes.items():
            content_dict = {
                ApiField.FIGURE_ID: str(figure_id),
                ApiField.GEOMETRY: (str(figure_id), figure_bytes, "application/sla"),
            }
            encoder = MultipartEncoder(fields=content_dict)
            resp = self._api.post("figures.bulk.upload.geometry", encoder)

    def upload_stl_meshes(
        self,
        volume_id: int,
        spatial_figures: List[VolumeFigure],
        key_id_map: KeyIdMap,
        interpolation_dir=None,
    ):
        """
        Upload existing interpolations or create on the fly and and add them to empty mesh figures.

        :param volume_id: VolumeFigure ID in Supervisely.
        :type volume_id: int
        :param spatial_figures: List of spatial figures to upload.
        :type spatial_figures: List[VolumeFigure]
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19371414
            project_id = 17215

            volume = api.volume.get_info_by_id(volume_id)

            key_id_map = sly.KeyIdMap()
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            vol_ann_json = api.volume.annotation.download(volume_id)
            id_to_paths = {}
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)
            sp_figures = vol_ann.spatial_figures

            res = volume_figure_api.upload_stl_meshes(volume_id, sp_figures, key_id_map)
        """

        if len(spatial_figures) == 0:
            return

        figure2bytes = {}
        for sp in spatial_figures:
            figure_id = key_id_map.get_figure_id(sp.key())
            if interpolation_dir is not None:
                meth_path = os.path.join(interpolation_dir, sp.key().hex + ".stl")
                if file_exists(meth_path):
                    with open(meth_path, "rb") as in_file:
                        meth_bytes = in_file.read()
                    figure2bytes[figure_id] = meth_bytes
            # else - no stl file
            if figure_id not in figure2bytes:
                meth_bytes = self.interpolate(volume_id, sp, key_id_map)
                figure2bytes[figure_id] = meth_bytes
        self._upload_meshes_batch(figure2bytes)

    def _append_bulk_mask3d(
        self,
        entity_id: int,
        figures_json: List,
        figures_keys: List,
        key_id_map: KeyIdMap,
        field_name=ApiField.ENTITY_ID,
    ):
        """The same method as _append_bulk but for spatial figures. Uploads figures to given Volume by ID.
        You need to upload the geometry right after figures will be created

        :param entity_id: Volume ID.
        :type entity_id: int
        :param figures_json: List of figure dicts.
        :type figures_json: list
        :param figures_keys: List of figure keys as UUID.
        :type figures_keys: list
        :param key_id_map: KeyIdMap object (dict with bidict values)
        :type key_id_map: KeyIdMap
        :param field_name: field name for request body
        :type field_name: str
        :rtype: :class:`NoneType`
        :Usage example:
        """
        figures_count = len(figures_json)
        if figures_count == 0:
            return

        fake_figures = []
        for figure in figures_json:
            fake_figures.append(
                {
                    "objectId": figure["objectId"],
                    "geometryType": Mask3D.name(),
                    "tool": Mask3D.name(),
                    "entityId": entity_id,
                }
            )
        for batch_keys, batch_jsons in zip(
            batched(figures_keys, batch_size=100), batched(fake_figures, batch_size=100)
        ):
            resp = self._api.post(
                "figures.bulk.add",
                {field_name: entity_id, ApiField.FIGURES: batch_jsons},
            )
            for key, resp_obj in zip(batch_keys, resp.json()):
                figure_id = resp_obj[ApiField.ID]
                key_id_map.add_figure(key, figure_id)

    def upload_sf_geometry(self, spatial_figures: Dict, geometries: List, key_id_map: KeyIdMap):
        """
        Upload spatial figures geometry as bytes to storage by given ID.

        :param spatial_figures: Dictionary with figure IDs.
        :type spatial_figures: dict
        :param geometries: Dictionary with geometries, which represented as content of NRRD files in byte format.
        :type geometries: list
        :param key_id_map: KeyIdMap object (dict with bidict values)
        :type key_id_map: KeyIdMap
        :rtype: :class:`NoneType`
        :Usage example:
        """

        for sf, geometry_bytes in zip(spatial_figures, geometries):
            figure_id = key_id_map.get_figure_id(sf.key())
            content_dict = {
                ApiField.FIGURE_ID: str(figure_id),
                ApiField.GEOMETRY: (str(figure_id), geometry_bytes, "application/sla"),
            }
            encoder = MultipartEncoder(fields=content_dict)
            self._api.post("figures.bulk.upload.geometry", encoder)
