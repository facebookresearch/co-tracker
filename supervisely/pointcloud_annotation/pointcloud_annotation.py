# coding: utf-8
from __future__ import annotations
from typing import Optional, List, Dict
from supervisely.project.project_meta import ProjectMeta
from copy import deepcopy
import uuid
import json

from supervisely._utils import take_with_default
from supervisely.pointcloud_annotation.pointcloud_tag_collection import PointcloudTagCollection
from supervisely.pointcloud_annotation.constants import (
    DESCRIPTION,
    TAGS,
    OBJECTS,
    KEY,
    FIGURES,
    POINTCLOUD_ID,
)
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)


class PointcloudAnnotation(VideoAnnotation):
    """
    Class for creating and using PointcloudAnnotation

    :param objects: PointcloudObjectCollection object
    :type objects: PointcloudObjectCollection, optional
    :param figures: List[PointcloudFigure] object
    :type figures: List[PointcloudFigure], optional
    :param tags: PointcloudTagCollection object
    :type tags: PointcloudTagCollection, optional
    :param description: Description text
    :type description: str, optional
    :param key: uuid class object
    :type key: uuid.UUID, optional

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        from supervisely.video_annotation.key_id_map import KeyIdMap

        # PointcloudAnnotation example 1
        pointcloud_ann = sly.PointcloudAnnotation()
        print(pointcloud_ann.to_json())
        # Output: {
        #     "description": "",
        #     "key": "ad97e8a4a8524b8a992d1f083c5e6b00",
        #     "tags": [],
        #     "objects": [],
        #     "figures": []
        # }


        # PointcloudAnnotation example 2
        key_id_map = KeyIdMap()
        project_meta_json = api.project.get_meta(pcd_info.project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta_json)
        ann_json = api.pointcloud.annotation.download(pointcloud_id)
        ann = sly.PointcloudAnnotation.from_json(
            data=ann_json, project_meta=project_meta, key_id_map=key_id_map
        )
    """

    def __init__(
        self,
        objects: Optional[PointcloudObjectCollection] = None,
        figures: Optional[List[PointcloudFigure]] = None,
        tags: Optional[PointcloudTagCollection] = None,
        description: Optional[str] = "",
        key: Optional[uuid.UUID] = None,
    ):

        self._description = description
        self._tags = take_with_default(tags, PointcloudTagCollection())
        self._objects = take_with_default(objects, PointcloudObjectCollection())
        self._figures = take_with_default(figures, [])
        self._key = take_with_default(key, uuid.uuid4())

    @property
    def img_size(self):
        """Not supported for pointcloud"""

        raise NotImplementedError("Not supported for pointcloud")

    @property
    def frames_count(self):
        """Not supported for pointcloud"""

        raise NotImplementedError("Not supported for pointcloud")

    @property
    def frames(self):
        """Not supported for pointcloud"""

        raise NotImplementedError("Not supported for pointcloud")

    @property
    def tags(self) -> PointcloudTagCollection:
        """
        PointcloudTag objects collection.

        :returns: PointcloudTagCollection object.
        :rtype: PointcloudTagCollection

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            project_id = 19441
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            # Load json file
            path = "/home/admin/work/docs/my_dataset/ann/annotation.json"
            ann = sly.PointcloudAnnotation.load_json_file(path, project_meta)

            tags = ann.tags
        """

        return super().tags

    @property
    def objects(self) -> PointcloudObjectCollection:
        """
        PointcloudObject objects collection.

        :returns: PointcloudObjectCollection object.
        :rtype: PointcloudObjectCollection

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            project_id = 19441
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            # Load json file
            path = "/home/admin/work/docs/my_dataset/ann/annotation.json"
            ann = sly.PointcloudAnnotation.load_json_file(path, project_meta)

            objects = ann.objects
        """

        return super().objects

    @property
    def figures(self) -> List[PointcloudFigure]:
        """
        PointcloudFigure objects.

        :returns: List of PointcloudFigure objects from PointcloudAnnotation object.
        :rtype: list

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            project_id = 19441
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            # Load json file
            path = "/home/admin/work/docs/my_dataset/ann/annotation.json"
            ann = sly.PointcloudAnnotation.load_json_file(path, project_meta)

            figures = ann.figures
        """

        return deepcopy(self._figures)

    # def get_objects_on_frame(self, frame_index: int):
    #     raise NotImplementedError("Not supported for pointcloud")

    # def get_tags_on_frame(self, frame_index: int):
    #     raise NotImplementedError("Not supported for pointcloud")

    def get_objects_from_figures(self) -> PointcloudObjectCollection:
        """
        Get PointcloudObjectCollection object from annotation figures.

        :return: PointcloudObjectCollection object from annotation figures.
        :rtype: PointcloudObjectCollection
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            key_id_map = KeyIdMap()
            project_id = 19441
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            ann_json = api.pointcloud.annotation.download(pointcloud_id)
            ann = sly.PointcloudAnnotation.from_json(
                data=ann_json, project_meta=project_meta, key_id_map=key_id_map
            )

            objects = ann.get_objects_from_figures()
        """

        ann_objects = {}
        for fig in self.figures:
            if fig.parent_object.key() not in ann_objects.keys():
                ann_objects[fig.parent_object.key()] = fig.parent_object

        return PointcloudObjectCollection(ann_objects.values())

    def validate_figures_bounds(self):
        """Not supported for pointcloud"""

        raise NotImplementedError("Not supported for pointcloud")

    def to_json(self, key_id_map: Optional[KeyIdMap] = None) -> Dict:
        """
        Convert PointcloudAnnotation to json format.

        :return: PointcloudAnnotation in json format
        :rtype: Dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            pointcloud_ann = sly.PointcloudAnnotation()

            print(pointcloud_ann.to_json())
            # Output: {
            #     "description": "",
            #     "key": "ad97e8a4a8524b8a992d1f083c5e6b00",
            #     "tags": [],
            #     "objects": [],
            #     "figures": []
            # }
        """

        res_json = {
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            FIGURES: [figure.to_json(key_id_map) for figure in self.figures],
        }

        if key_id_map is not None:
            pointcloud_id = key_id_map.get_video_id(self.key())
            if pointcloud_id is not None:
                res_json[POINTCLOUD_ID] = pointcloud_id

        return res_json

    @classmethod
    def from_json(
        cls, data: Dict, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudAnnotation:
        """
        Convert pointcloud annotation from json format in PointcloudAnnotation object.

        :param data: Pointcloud annotation in json format.
        :type data: Dict
        :param project_meta: Project metadata.
        :type project_meta: ProjectMeta
        :return: PointcloudAnnotation object.
        :rtype: PointcloudAnnotation
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.key_id_map import KeyIdMap

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            key_id_map = KeyIdMap()
            pointcloud_id = pointcloud_id
            project_id = 19441
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            ann_json = api.pointcloud.annotation.download(pointcloud_id)

            ann = sly.PointcloudAnnotation.from_json(
                data=ann_json, project_meta=project_meta, key_id_map=key_id_map
            )
        """

        try:
            item_key = uuid.UUID(data[KEY])
        except Exception as e:
            item_key = uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(item_key, data.get(POINTCLOUD_ID, None))
        description = data.get(DESCRIPTION, "")
        tags = PointcloudTagCollection.from_json(data[TAGS], project_meta.tag_metas, key_id_map)
        objects = PointcloudObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)

        figures = []
        for figure_json in data.get(FIGURES, []):
            figure = PointcloudFigure.from_json(figure_json, objects, None, key_id_map)
            figures.append(figure)

        return cls(
            objects=objects, figures=figures, tags=tags, description=description, key=item_key
        )

    @classmethod
    def load_json_file(
        cls, path: str, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudAnnotation:
        """
        Loads json file and converts it to PointcloudAnnotation.

        :param path: Path to the json file.
        :type path: str
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudAnnotation object
        :rtype: :class:`PointcloudAnnotation<PointcloudAnnotation>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            team_name = 'Vehicle Detection'
            workspace_name = 'Cities'
            project_name =  'London'

            team = api.team.get_info_by_name(team_name)
            workspace = api.workspace.get_info_by_name(team.id, workspace_name)
            project = api.project.get_info_by_name(workspace.id, project_name)

            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            # Load json file
            path = "/home/admin/work/docs/my_dataset/ann/annotation.json"
            ann = sly.PointcloudAnnotation.load_json_file(path, project_meta)
        """

        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta, key_id_map)

    def clone(
        self,
        objects: Optional[PointcloudObjectCollection] = None,
        figures: Optional[List] = None,
        tags: Optional[PointcloudTagCollection] = None,
        description: Optional[str] = None,
    ) -> PointcloudAnnotation:
        """
        Makes a copy of PointcloudAnnotation with new fields, if fields are given, otherwise it will use fields of the original PointcloudAnnotation.

        :param objects: PointcloudObjectCollection object
        :type objects: PointcloudObjectCollection
        :param figures: list of pointcloud figures
        :type figures: list of figures
        :param tags: PointcloudTagCollection object
        :type tags: PointcloudTagCollection
        :param description: Description text
        :type description: str
        :return: PointcloudAnnotation class object

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.key_id_map import KeyIdMap

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            key_id_map = KeyIdMap()
            pointcloud_id = 19481098
            project_id = 19441
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            ann_json = api.pointcloud.annotation.download(pointcloud_id)
            ann = sly.PointcloudAnnotation.from_json(
                data=ann_json, project_meta=project_meta, key_id_map=key_id_map
            )

            obj_class_car = sly.ObjClass('car', sly.Cuboid)
            pointcloud_obj_car = sly.PointcloudObject(obj_class_car)
            new_objects = sly.PointcloudObjectCollection([pointcloud_obj_car])

            new_ann = ann.clone(objects=new_objects)

            print(new_ann.to_json())
            # Output: {
            #     "description": "",
            #     "figures": [],
            #     "key": "2cc443272aca4cfa9c4f404614938aa7",
            #     "objects": [
            #         {
            #         "classTitle": "Pole",
            #         "createdAt": "2023-03-16T06:38:44.934Z",
            #         "key": "eff2ec5e3cda47968f45bc51b36a0dc1",
            #         "labelerLogin": "almaz",
            #         "tags": [],
            #         "updatedAt": "2023-03-16T06:38:44.934Z"
            #         },
            #         {
            #         "classTitle": "Tram",
            #         "createdAt": "2023-03-16T06:38:44.934Z",
            #         "key": "6baa92e09ceb413ba8fbfcfae74be1c7",
            #         "labelerLogin": "almaz",
            #         "tags": [],
            #         "updatedAt": "2023-03-16T06:38:44.934Z"
            #         },
            #         {
            #         "classTitle": "car",
            #         "key": "6b1bced23061437b8ddbcdd267548c96",
            #         "tags": []
            #         }
            #     ],
            #     "tags": []
            # }
        """

        return PointcloudAnnotation(
            objects=take_with_default(objects, self.objects),
            figures=take_with_default(figures, self.figures),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
        )
