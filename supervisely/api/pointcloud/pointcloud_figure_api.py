# coding: utf-8

from typing import Dict, List, Optional
from supervisely.api.entity_annotation.figure_api import FigureApi, ApiField
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.video_annotation.key_id_map import KeyIdMap


class PointcloudFigureApi(FigureApi):
    """
    :class:`PointcloudFigure<supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure>` for a single point cloud.
    """

    def create(
        self,
        pointcloud_id: int,
        object_id: int,
        geometry_json: Dict,
        geometry_type: str,
        track_id: Optional[int] = None,
    ) -> int:
        """
        Create new PointcloudFigure of given point cloud object in point cloud with given ID.

        :param pointcloud_id: Point cloud ID in Supervisely.
        :type pointcloud_id: int
        :param object_id: ID of the object to which the PointcloudFigure belongs.
        :type object_id: int
        :param geometry_json: Parameters of geometry for PointcloudFigure.
        :type geometry_json: dict
        :param geometry_type: Type of PointcloudFigure geometry.
        :type geometry_type: str
        :param track_id: int, optional.
        :type track_id: int, optional
        :return: New figure ID
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pcd_id = 19618685
            object_id = 5565921
            geometry_json = {'points': {'exterior': [[500, 500], [1555, 1500]], 'interior': []}}
            geometry_type = 'rectangle'

            figure_id = api.pointcloud.figure.create(pcd_id, object_id, geometry_json, geometry_type) # 643182610
        """

        return super().create(pointcloud_id, object_id, {}, geometry_json, geometry_type, track_id)

    def append_bulk(
        self,
        pointcloud_id: int,
        figures: List[PointcloudFigure],
        key_id_map: KeyIdMap,
    ) -> None:
        """
        Add VideoFigures to given Video by ID.

        :param pointcloud_id: Point cloud ID in Supervisely.
        :type pointcloud_id: int
        :param figures: List of point cloud figures to append.
        :type figures: List[PointcloudFigure]
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

            project_id = 124976
            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)
            key_id_map = KeyIdMap()

            pcd_id = 198703212
            ann_info = api.pointcloud.annotation.download(pcd_id)
            ann = sly.PointcloudAnnotation.from_json(ann_info, meta, key_id_map)
            figures = ann.figures[:5]
            api.video.figure.append_bulk(pcd_id, figures, key_id_map)
        """

        keys = []
        figures_json = []
        for figure in figures:
            keys.append(figure.key())
            figures_json.append(figure.to_json(key_id_map))

        self._append_bulk(pointcloud_id, figures_json, keys, key_id_map)

    def append_to_dataset(
        self,
        dataset_id: int,
        figures: List[PointcloudFigure],
        entity_ids: List[int],
        key_id_map: KeyIdMap,
    ) -> None:
        """
        Add pointcloud figures to Dataset annotations.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :param figures: List of point cloud figures.
        :type figures: List[PointcloudFigure]
        :param entity_ids: List of point cloud IDs.
        :type entity_ids: List[int]
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :rtype: :class:`NoneType`
        :Usage example:

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
            from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudObjectCollection
            from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
            from supervisely.video_annotation.key_id_map import KeyIdMap

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()


            project_id = 17231
            dataset_id = 55875
            pointcloud_id = 19373403
            project = api.project.get_info_by_id(project_id)
            dataset = api.dataset.get_info_by_id(dataset_id)

            class_car = sly.ObjClass('car', Cuboid3d)
            classes = sly.ObjClassCollection([class_car])
            project_meta = sly.ProjectMeta(classes)
            updated_meta = api.project.update_meta(project.id, project_meta.to_json())

            key_id_map = KeyIdMap()

            car_object = sly.PointcloudObject(class_car)
            objects_collection = PointcloudObjectCollection([car_object])

            uploaded_objects_ids = api.pointcloud_episode.object.append_to_dataset(
                dataset.id,
                objects_collection,
                key_id_map,
            )

            position, rotation, dimension = Vector3d(-32.4, 33.9, -0.7), Vector3d(0., 0, 0.1), Vector3d(1.8, 3.9, 1.6)
            cuboid = Cuboid3d(position, rotation, dimension)
            figure_1 = PointcloudFigure(car_object, cuboid)

            api.pointcloud_episode.figure.append_to_dataset(
                dataset.id,
                [figure_1],
                [pointcloud_id],
                key_id_map,
            )
        """

        keys = []
        figures_json = []
        for figure, entity_id in zip(figures, entity_ids):
            keys.append(figure.key())
            figure_json = figure.to_json(key_id_map)
            figure_json[ApiField.ENTITY_ID] = entity_id
            figures_json.append(figure_json)

        return self._append_bulk(
            dataset_id, figures_json, keys, key_id_map, field_name=ApiField.DATASET_ID
        )

    def _convert_json_info(self, info: dict, skip_missing=True):
        return super()._convert_json_info(info, skip_missing)
