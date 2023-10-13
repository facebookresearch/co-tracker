# coding: utf-8
from __future__ import annotations
import uuid
import json
from typing import Optional, Dict, List

from supervisely.project.project_meta import ProjectMeta
from supervisely._utils import take_with_default
from supervisely.api.module_api import ApiField
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.video_annotation.constants import (
    FRAMES,
    DESCRIPTION,
    FRAMES_COUNT,
    TAGS,
    OBJECTS,
    KEY,
)
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_episode_frame_collection import (
    PointcloudEpisodeFrameCollection,
)
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_episode_tag_collection import (
    PointcloudEpisodeTagCollection,
)


class PointcloudEpisodeAnnotation:
    """
    PointcloudEpisodeAnnotation for point cloud episodes.
    :class:`PointcloudEpisodeAnnotation<PointcloudEpisodeAnnotation>` object is immutable.

    :param frames_count: Number of PointcloudEpisodeFrame objects.
    :type frames_count: int, optional
    :param objects: PointcloudObjectCollection object
    :type objects: PointcloudObjectCollection, optional
    :param frames: PointcloudEpisodeFrameCollection object
    :type frames: PointcloudEpisodeFrameCollection, optional
    :param tags: PointcloudEpisodeTagCollection object
    :type tags: PointcloudEpisodeTagCollection, optional
    :param description: Description text
    :type description: str, optional
    :param key: uuid class object
    :type key: uuid.UUID, optional

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        from supervisely.video_annotation.key_id_map import KeyIdMap

        # PointcloudEpisodeAnnotation example 1
        pointcloud_episodes_ann = sly.PointcloudEpisodeAnnotation()
        print(pointcloud_episodes_ann.to_json())
        # Output: {
        #     "description": "",
        #     "frames": [],
        #     "framesCount": None,
        #     "key": "494f67984d714c1eaf7a65e5df289ac6",
        #     "objects": [],
        #     "tags": []
        # }


        # PointcloudEpisodeAnnotation example 2
        pointcloud_id = 19481098
        key_id_map = KeyIdMap()
        pcd_info = api.pointcloud_episode.get_info_by_id(pointcloud_id)
        project_meta_json = api.project.get_meta(pcd_info.project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta_json)
        ann_json = api.pointcloud_episode.annotation.download(pcd_info.dataset_id)
        ann = sly.PointcloudEpisodeAnnotation.from_json(
            data=ann_json, project_meta=project_meta, key_id_map=key_id_map
        )
    """

    def __init__(
        self,
        frames_count: Optional[int] = None,
        objects: Optional[PointcloudObjectCollection] = None,
        frames: Optional[PointcloudEpisodeFrameCollection] = None,
        tags: Optional[PointcloudEpisodeTagCollection] = None,
        description: Optional[str] = "",
        key: uuid.UUID = None,
    ) -> None:
        self._frames_count = frames_count
        self._description = description
        self._frames = take_with_default(frames, PointcloudEpisodeFrameCollection())
        self._tags = take_with_default(tags, PointcloudEpisodeTagCollection())
        self._objects = take_with_default(objects, PointcloudObjectCollection())
        self._key = take_with_default(key, uuid.uuid4())

    def get_tags_on_frame(self, frame_index: int) -> PointcloudEpisodeTagCollection:
        """
        Retrieve tags associated with a specific frame in a PointcloudEpisodeAnnotation.

        :param frame_index: The index of the frame for which tags need to be retrieved.
        :type frame_index: int
        :return: PointcloudEpisodeTagCollection containing the retrieved tags associated with the specified frame.
        :rtype: PointcloudEpisodeTagCollection
        :raises ValueError: If no frame with the given frame_index exists in the annotation.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.key_id_map import KeyIdMap

            key_id_map = KeyIdMap()
            pointcloud_id = 19481098
            pcd_info = api.pointcloud_episode.get_info_by_id(pointcloud_id)
            project_meta_json = api.project.get_meta(pcd_info.project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            ann_json = api.pointcloud_episode.annotation.download(pcd_info.dataset_id)
            ann = sly.PointcloudEpisodeAnnotation.from_json(
                data=ann_json, project_meta=project_meta, key_id_map=key_id_map
            )
            frame_index = 0
            tags_on_frame = ann.get_tags_on_frame(frame_index)

            print(tags_on_frame)
            # Output:
            Tags:
            +-------+------------+-------+-------------+
            |  Name | Value type | Value | Frame range |
            +-------+------------+-------+-------------+
            | color | any_string |  red  |   [0, 0]   |
            +-------+------------+-------+-------------+
        """

        frame = self._frames.get(frame_index, None)
        if frame is None:
            if frame_index < self.frames_count:
                return PointcloudEpisodeTagCollection([])
            else:
                raise ValueError(f"No frame with index {frame_index} in annotation.")
        tags = []
        for tag in self._tags:
            if frame_index >= tag.frame_range[0] and frame_index <= tag.frame_range[1]:
                tags.append(tag)
        return PointcloudEpisodeTagCollection(tags)

    def get_objects_on_frame(self, frame_index: int) -> PointcloudObjectCollection:
        """
        Retrieve objects associated with a specific frame in a PointcloudEpisodeAnnotation.

        :param frame_index: The index of the frame for which objects need to be retrieved.
        :type frame_index: int
        :return: PointcloudObjectCollection containing the retrieved objects associated with the specified frame.
        :rtype:
        :raises ValueError: If no frame with the given frame_index exists in the annotation.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.key_id_map import KeyIdMap

            key_id_map = KeyIdMap()
            pointcloud_id = 19481098
            pcd_info = api.pointcloud_episode.get_info_by_id(pointcloud_id)
            project_meta_json = api.project.get_meta(pcd_info.project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            ann_json = api.pointcloud_episode.annotation.download(pcd_info.dataset_id)
            ann = sly.PointcloudEpisodeAnnotation.from_json(
                data=ann_json, project_meta=project_meta, key_id_map=key_id_map
            )
            frame_index = 0
            objects_on_frame = ann.get_objects_on_frame(frame_index)

            print(objects_on_frame.to_json())
            # Output:
            # [
            #     {
            #         "key": "687784c3d4d64ec4811948fec245514a",
            #         "classTitle": "Tram",
            #         "tags": [],
            #         "labelerLogin": "almaz",
            #         "updatedAt": "2023-03-16T06:38:44.934Z",
            #         "createdAt": "2023-03-16T06:38:44.934Z"
            #     },
            #     {
            #         "key": "b8b23b6712444f0fbfb320b0b4acd09a",
            #         "classTitle": "Car",
            #         "tags": [],
            #         "labelerLogin": "almaz",
            #         "updatedAt": "2023-03-16T06:38:44.934Z",
            #         "createdAt": "2023-03-16T06:38:44.934Z"
            #     }
            # ]

        """

        frame = self._frames.get(frame_index, None)
        if frame is None:
            if frame_index < self.frames_count:
                return PointcloudObjectCollection([])
            else:
                raise ValueError(f"No frame with index {frame_index} in annotation.")
        frame_objects = {}
        for fig in frame.figures:
            if fig.parent_object.key() not in frame_objects.keys():
                frame_objects[fig.parent_object.key()] = fig.parent_object
        return PointcloudObjectCollection(list(frame_objects.values()))

    def get_figures_on_frame(self, frame_index: int) -> List[PointcloudFigure]:
        """
        Retrieve figures associated with a specific frame in a PointcloudEpisodeAnnotation.

        :param frame_index: The index of the frame for which figures need to be retrieved.
        :type frame_index: int
        :return: List of PointcloudFigure objects containing the retrieved figures associated with the specified frame.
        :rtype: List[PointcloudFigure]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.key_id_map import KeyIdMap

            key_id_map = KeyIdMap()
            pointcloud_id = 19481098
            pcd_info = api.pointcloud_episode.get_info_by_id(pointcloud_id)
            project_meta_json = api.project.get_meta(pcd_info.project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            ann_json = api.pointcloud_episode.annotation.download(pcd_info.dataset_id)
            ann = sly.PointcloudEpisodeAnnotation.from_json(
                data=ann_json, project_meta=project_meta, key_id_map=key_id_map
            )
            frame_index = 0
            figures_on_frame = ann.get_figures_on_frame(frame_index)

            print(figures_on_frame)
            # Output:
            # [<supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure object at 0x7fc83895a4d0>,
            # <supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure object at 0x7fc83895a810>,
            # <supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure object at 0x7fc8389410d0>]
        """

        frame = self._frames.get(frame_index, None)
        if frame is None:
            if frame_index < self.frames_count:
                return PointcloudObjectCollection([])
            else:
                raise ValueError(f"No frame with index {frame_index} in annotation.")
        return frame.figures

    def to_json(self, key_id_map: KeyIdMap = None) -> Dict:
        """
        Convert PointcloudEpisodeAnnotation to JSON format.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudEpisodeAnnotation in JSON format.
        :rtype: Dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            pointcloud_id = 19481098
            pcd_info = api.pointcloud_episode.get_info_by_id(pointcloud_id)
            project_path = "Downloads/pointcloud_api/project"
            sly.PointcloudEpisodeProject.download(
                api=api,
                project_id=pcd_info.project_id,
                dest_dir=project_path,
                dataset_ids=[pcd_info.dataset_id],
                download_pointclouds=True,
            )
            project_fs = sly.PointcloudEpisodeProject(project_path, sly.OpenMode.READ)
            project_meta_json = api.project.get_meta(pcd_info.project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            ds = project_fs.datasets.items()[0]:
            ann = ds.get_ann(project_meta)

            print(ann.to_json())
            # Output:
            # {
            #     'datasetId': 60988,
            #     'description': '',
            #     'frames': [{'figures': [{'classId': None,
            #                             'createdAt': '2023-03-16T06:38:45.004Z',
            #                             'description': '',
            #                             'geometry': {'dimensions': {'x': 2.3652234,
            #                                                         'y': 23.291742,
            #                                                         'z': 3.326648},
            #                                         'position': {'x': 86.29707472161449,
            #                                                         'y': -14.472597682830635,
            #                                                         'z': 0.8842007608554671},
            #                                         'rotation': {'x': 0,
            #                                                         'y': 0,
            #                                                         'z': -1.6962800995995606}},
            #                             'geometryType': 'cuboid_3d',
            #                             'id': 87536496,
            #                             'labelerLogin': 'almaz',
            #                             'objectId': 5531328,
            #                             'updatedAt': '2023-03-16T06:38:45.004Z'}],
            #                 'index': 0,
            #                 'pointCloudId': 19481098}],
            #                 'index': 1,
            #                 'pointCloudId': 19481100},
            #                 ...],
            #     'framesCount': 54,
            #     'objects': [{'classId': 666944,
            #                 'classTitle': 'Car',
            #                 'createdAt': '2023-03-16T06:38:44.934Z',
            #                 'datasetId': 60988,
            #                 'entityId': None,
            #                 'id': 5531324,
            #                 'labelerLogin': 'almaz',
            #                 'tags': [],
            #                 'updatedAt': '2023-03-16T06:38:44.934Z'}],
            #     'tags': []
            # }
        """

        res_json = {
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            FRAMES_COUNT: self.frames_count,
            FRAMES: self.frames.to_json(key_id_map),
        }

        if key_id_map is not None:
            dataset_id = key_id_map.get_video_id(self.key())
            if dataset_id is not None:
                res_json[ApiField.DATASET_ID] = dataset_id

        return res_json

    @classmethod
    def from_json(
        cls, data: Dict, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudEpisodeAnnotation:
        """
        Create a PointcloudEpisodeAnnotation object from a JSON representation.

        :param data: JSON data representing the PointcloudEpisodeAnnotation.
        :type data: Dict
        :param project_meta: Project metadata.
        :type project_meta: ProjectMeta
        :return: PointcloudEpisodeAnnotation object
        :rtype: :class:`PointcloudEpisodeAnnotation<PointcloudEpisodeAnnotation>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.key_id_map import KeyIdMap

            key_id_map = KeyIdMap()
            pointcloud_id = 19481098
            pcd_info = api.pointcloud_episode.get_info_by_id(pointcloud_id)
            project_meta_json = api.project.get_meta(pcd_info.project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            ann_json = api.pointcloud_episode.annotation.download(pcd_info.dataset_id)

            ann = sly.PointcloudEpisodeAnnotation.from_json(
                data=ann_json, project_meta=project_meta, key_id_map=key_id_map
            )
        """

        item_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(item_key, data.get(ApiField.DATASET_ID, None))

        description = data.get(DESCRIPTION, "")
        frames_count = data.get(FRAMES_COUNT, 0)

        tags = PointcloudEpisodeTagCollection.from_json(
            data[TAGS], project_meta.tag_metas, key_id_map
        )
        objects = PointcloudObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)
        frames = PointcloudEpisodeFrameCollection.from_json(
            data[FRAMES], objects, key_id_map=key_id_map
        )

        return cls(frames_count, objects, frames, tags, description, item_key)

    @classmethod
    def load_json_file(
        cls, path: str, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudEpisodeAnnotation:
        """
        Loads json file and converts it to PointcloudEpisodeAnnotation.

        :param path: Path to the json file.
        :type path: str
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudEpisodeAnnotation object
        :rtype: :class:`PointcloudEpisodeAnnotation<PointcloudEpisodeAnnotation>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            team_name = 'Vehicle Detection'
            workspace_name = 'Cities'
            project_name =  'London'

            project_id = 19441
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            # Load json file
            path = "/home/admin/work/docs/my_dataset/ann/annotation.json"
            ann = sly.PointcloudEpisodeAnnotation.load_json_file(path, project_meta)
        """
        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta, key_id_map)

    def clone(
        self,
        frames_count: Optional[int] = None,
        objects: Optional[PointcloudObjectCollection] = None,
        frames: Optional[PointcloudEpisodeFrameCollection] = None,
        tags: Optional[PointcloudEpisodeTagCollection] = None,
        description: Optional[str] = "",
    ) -> PointcloudEpisodeAnnotation:
        """
        Makes a copy of PointcloudEpisodeAnnotation with new fields, if fields are given, otherwise it will use fields of the original PointcloudEpisodeAnnotation.

        :param frames_count: Number of PointcloudEpisodeFrame objects
        :type frames_count: int, optional
        :param objects: PointcloudObjectCollection object
        :type objects: PointcloudObjectCollection, optional
        :param frames: PointcloudEpisodeFrameCollection object
        :type frames: PointcloudEpisodeFrameCollection, optional
        :param tags: PointcloudEpisodeTagCollection object
        :type tags: PointcloudEpisodeTagCollection, optional
        :param description: Description text
        :type description: str, optional
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
            pcd_info = api.pointcloud_episode.get_info_by_id(pointcloud_id)
            project_meta_json = api.project.get_meta(pcd_info.project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            ann_json = api.pointcloud_episode.annotation.download(pcd_info.dataset_id)
            ann = sly.PointcloudAnnotation.from_json(
                data=ann_json, project_meta=project_meta, key_id_map=key_id_map
            )

            obj_class_car = sly.ObjClass('car', sly.Cuboid)
            pointcloud_obj_car = sly.PointcloudObject(obj_class_car)
            new_objects = ann.objects.add(pointcloud_obj_car)

            new_ann = ann.clone(objects=new_objects)

            print(new_ann.to_json())
            # Output:
            # {
            #     'datasetId': 60988,
            #     'description': '',
            #     'frames': [{'figures': [{'classId': None,
            #                             'createdAt': '2023-03-16T06:38:45.004Z',
            #                             'description': '',
            #                             'geometry': {'dimensions': {'x': 2.3652234,
            #                                                         'y': 23.291742,
            #                                                         'z': 3.326648},
            #                                         'position': {'x': 86.29707472161449,
            #                                                         'y': -14.472597682830635,
            #                                                         'z': 0.8842007608554671},
            #                                         'rotation': {'x': 0,
            #                                                         'y': 0,
            #                                                         'z': -1.6962800995995606}},
            #                             'geometryType': 'cuboid_3d',
            #                             'id': 87536496,
            #                             'labelerLogin': 'almaz',
            #                             'objectId': 5531328,
            #                             'updatedAt': '2023-03-16T06:38:45.004Z'}],
            #                 'index': 0,
            #                 'pointCloudId': 19481098}],
            #                 'index': 1,
            #                 'pointCloudId': 19481100},
            #                 ...],
            #     'framesCount': 54,
            #     'objects': [{
            #                     'classId': 666944,
            #                     'classTitle': 'Car',
            #                     'createdAt': '2023-03-16T06:38:44.934Z',
            #                     'datasetId': 60988,
            #                     'entityId': None,
            #                     'id': 5531324,
            #                     'labelerLogin': 'almaz',
            #                     'tags': [],
            #                     'updatedAt': '2023-03-16T06:38:44.934Z'}
            #                 {
            #                     'classTitle': 'Car',
            #                     'createdAt': '2023-03-16T06:38:44.934Z',
            #                     'key': 'fc149a8f3e3a413c807a6b4ba474645c',
            #                     'labelerLogin': 'almaz',
            #                     'tags': [],
            #                     'updatedAt': '2023-03-16T06:38:44.934Z
            #                 }],
            #     'tags': []
            # }
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

        return PointcloudEpisodeAnnotation(
            frames_count=take_with_default(frames_count, self.frames_count),
            objects=take_with_default(objects, self.objects),
            frames=take_with_default(frames, self.frames),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
        )

    @property
    def frames_count(self) -> int:
        """
        Number of frames.

        :return: Frames count
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            frames_count = 15
            video_ann = sly.PointcloudEpisodeAnnotation(frames_count=frames_count)
            print(video_ann.frames_count)
            # Output: 15
        """

        return self._frames_count

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
            ann = sly.PointcloudEpisodeAnnotation.load_json_file(path, project_meta)

            objects = ann.objects
        """

        return self._objects

    @property
    def frames(self) -> PointcloudEpisodeFrameCollection:
        """
        PointcloudEpisodeFrameCollection collection.

        :return: PointcloudEpisodeFrameCollection object
        :rtype: :class:`PointcloudEpisodeFrameCollection<supervisely.pointcloud_episodes.pointcloud_episode_frame_collection.PointcloudEpisodeFrameCollection>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
            from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

            obj_class_car = sly.ObjClass('car', Cuboid3d)
            pointcloud_obj_car = sly.PointcloudObject(obj_class_car)
            objects = sly.PointcloudObjectCollection([pointcloud_obj_car])

            position, rotation, dimension = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
            cuboid = Cuboid3d(position, rotation, dimension)
            frame_index = 10
            figure = sly.PointcloudFigure(pointcloud_obj_car, cuboid, frame_index=frame_index)
            frame = sly.PointcloudEpisodeFrame(frame_index, figures=[figure])
            frames = sly.PointcloudEpisodeFrameCollection([frame])

            pointcloud_episodes_ann = sly.PointcloudEpisodeAnnotation(frames_count, objects, frames)
            print(pointcloud_episodes_ann.frames.to_json())
            # Output:
            # [
            #     {
            #         "figures": [
            #         {
            #             "geometry": {
            #             "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
            #             "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
            #             "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
            #             },
            #             "geometryType": "cuboid_3d",
            #             "key": "030b9aafa97642e887e2be544ef7a7ee",
            #             "objectKey": "95c473a6cff44afda127ffb40d2bac5b"
            #         }
            #         ],
            #         "index": 0
            #     }
            # ]
        """

        return self._frames

    @property
    def figures(self) -> List[PointcloudFigure]:
        """
        PointcloudFigure objects.

        :returns: List of PointcloudFigure objects from PointcloudEpisodeAnnotation object.
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
            ann = sly.PointcloudEpisodeAnnotation.load_json_file(path, project_meta)

            for figure in ann.figures:
                print(figure.to_json())

            # Output:
            # {
            #     "geometry": {
            #         "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
            #         "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
            #         "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
            #     },
            #     "geometryType": "cuboid_3d",
            #     "key": "01836c294c514250a11889f56cf210e9",
            #     "objectKey": "737c4df19c0c4cccbc48cf69b72abe36"
            # }
        """

        return self.frames.figures

    @property
    def tags(self) -> PointcloudEpisodeTagCollection:
        """
        PointcloudEpisodeTag objects collection.

        :returns: PointcloudEpisodeTagCollection object.
        :rtype: PointcloudEpisodeTagCollection

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
            ann = sly.PointcloudEpisodeAnnotation.load_json_file(path, project_meta)

            tags = ann.tags
        """

        return self._tags

    def key(self) -> uuid.UUID:
        """
        PointcloudEpisodeAnnotation key value.

        :returns: Key value of point cloud episodes annotation object.
        :rtype: str

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            ann = sly.PointcloudEpisodeAnnotation()

            print(ann.key())
            # Output: 93ab6292-c661-4a53-b407-85ed34f5b68a'
        """

        return self._key

    @property
    def description(self) -> str:
        """
        Description text for PointcloudEpisodeAnnotation object.

        :return: PointcloudEpisodeAnnotation description
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            descr = 'example'
            ann = sly.PointcloudEpisodeAnnotation(description=descr)
            print(ann.description) # example
        """

        return self._description

    def is_empty(self) -> bool:
        """
        Check whether point cloud episodes annotation contains objects or tags, or not.

        :returns: True if point cloud episodes annotation  is empty, False otherwise.
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.key_id_map import KeyIdMap

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            project_id = 18428
            dataset_id = 60988
            key_id_map = KeyIdMap()
            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_json = api.pointcloud_episode.annotation.download(dataset_id)
            ann = sly.PointcloudEpisodeAnnotation.from_json(ann_json, meta, key_id_map)

            print(ann.is_empty()) # False
        """

        if len(self.objects) == 0 and len(self.tags) == 0:
            return True
        else:
            return False
