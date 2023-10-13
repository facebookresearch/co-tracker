from __future__ import annotations
from typing import List, Dict, Optional, Any, Iterator

from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_episode_frame import PointcloudEpisodeFrame
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

class PointcloudEpisodeFrameCollection(FrameCollection):
    """
    Collection with :class:`PointcloudEpisodeFrame<supervisely.pointcloud_annotation.pointcloud_episode_frame.PointcloudEpisodeFrame>` instances. :class:`PointcloudEpisodeFrameCollection<PointcloudEpisodeFrameCollection>` object is immutable.

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
        from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

        # Create pointcloud object
        obj_class_car = sly.ObjClass('car', Cuboid3d)
        pointcloud_obj_car = sly.PointcloudObject(obj_class_car)
        objects = PointcloudObjectCollection([pointcloud_obj_car])

        # Create two figures
        frame_index_1 = 7
        position_1, rotation_1, dimension_1 = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
        cuboid_1 = Cuboid3d(position_1, rotation_1, dimension_1)

        frame_index_2 = 10
        position_2, rotation_2, dimension_2 = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
        cuboid_2 = Cuboid3d(position_2, rotation_2, dimension_2)

        figure_1 = sly.PointcloudFigure(pointcloud_obj_car, cuboid_1, frame_index=frame_index_1)
        figure_2 = sly.PointcloudFigure(pointcloud_obj_car, cuboid_2, frame_index=frame_index_2)

        # Create two frames for collection
        frame_1 = sly.PointcloudEpisodeFrame(frame_index_1, figures=[figure_1])
        frame_2 = sly.PointcloudEpisodeFrame(frame_index_2, figures=[figure_2])

        # Create PointcloudEpisodeFrameCollection
        pcd_episodes_fr_collection = sly.PointcloudEpisodeFrameCollection([frame_1, frame_2])

        print(pcd_episodes_fr_collection.to_json())
        # Output: [
        #     {
        #         "figures": [
        #         {
        #             "geometry": {
        #             "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
        #             "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
        #             "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
        #             },
        #             "geometryType": "cuboid_3d",
        #             "key": "c9fb727a9b53432fa0316d0a5b6043bc",
        #             "objectKey": "0fc681681b1f4b12909ccf685c53b43e"
        #         }
        #         ],
        #         "index": 7
        #     },
        #     {
        #         "figures": [
        #         {
        #             "geometry": {
        #             "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
        #             "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
        #             "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
        #             },
        #             "geometryType": "cuboid_3d",
        #             "key": "c3df69c90fa14cf284906ebad1c360ae",
        #             "objectKey": "0fc681681b1f4b12909ccf685c53b43e"
        #         }
        #         ],
        #         "index": 10
        #     }
        # ]


        # Add item to PointcloudEpisodeFrameCollection
        frame_index_3 = 13
        frame_3 = sly.PointcloudEpisodeFrame(frame_index_3)
        # Remember that PointcloudEpisodeFrameCollection is immutable, and we need to assign new instance of PointcloudEpisodeFrameCollection to a new variable
        new_pcd_episodes_fr_collection = pcd_episodes_fr_collection.add(frame_3)
        print(new_pcd_episodes_fr_collection.to_json())
        # Output: [
        #     {
        #         "figures": [
        #         {
        #             "geometry": {
        #             "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
        #             "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
        #             "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
        #             },
        #             "geometryType": "cuboid_3d",
        #             "key": "d16c2bf646664007bd22bcf2996710ed",
        #             "objectKey": "e09431246cc24231a9d76b9ba55ce4e7"
        #         }
        #         ],
        #         "index": 7
        #     },
        #     {
        #         "figures": [
        #         {
        #             "geometry": {
        #             "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
        #             "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
        #             "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
        #             },
        #             "geometryType": "cuboid_3d",
        #             "key": "9f915e2d4342408ab0d4517f485f9950",
        #             "objectKey": "e09431246cc24231a9d76b9ba55ce4e7"
        #         }
        #         ],
        #         "index": 10
        #     },
        #     { "figures": [], "index": 13 }
        # ]


        # You can also add multiple items to collection
        frame_3 = sly.PointcloudEpisodeFrame(12)
        frame_4 = sly.PointcloudEpisodeFrame(15)
        # Remember that PointcloudEpisodeFrameCollection is immutable, and we need to assign new instance of PointcloudEpisodeFrameCollection to a new variable
        new_pcd_episodes_fr_collection = pcd_episodes_fr_collection.add_items([frame_3, frame_4])
        print(new_pcd_episodes_fr_collection.to_json())
        # Output: [
        #     {
        #         "figures": [
        #         {
        #             "geometry": {
        #             "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
        #             "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
        #             "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
        #             },
        #             "geometryType": "cuboid_3d",
        #             "key": "df14bb10e76c4b4093a137f4ea01e3ba",
        #             "objectKey": "a83a21ef7b734bb9949ddf132b42d1f0"
        #         }
        #         ],
        #         "index": 7
        #     },
        #     {
        #         "figures": [
        #         {
        #             "geometry": {
        #             "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
        #             "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
        #             "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
        #             },
        #             "geometryType": "cuboid_3d",
        #             "key": "6f86a1859da34415a37896ec2c972b47",
        #             "objectKey": "a83a21ef7b734bb9949ddf132b42d1f0"
        #         }
        #         ],
        #         "index": 10
        #     },
        #     { "figures": [], "index": 12 },
        #     { "figures": [], "index": 15 }
        # ]


        # Has key, checks if given key exist in point cloud episodes frame collection
        pcd_episodes_fr_collection.has_key(10) # True
        pcd_episodes_fr_collection.has_key(22) # False


        # Intersection, finds intersection of given list of instances with collection items
        frame_1 = sly.PointcloudEpisodeFrame(1)
        frame_2 = sly.PointcloudEpisodeFrame(2)
        pcd_episodes_fr_collection = sly.PointcloudEpisodeFrameCollection([frame_1, frame_2])

        frame_3 = sly.PointcloudEpisodeFrame(3)

        frames_intersections = pcd_episodes_fr_collection.intersection([frame_3])
        print(frames_intersections.to_json())
        # Output: []

        frames_intersections = pcd_episodes_fr_collection.intersection([frame_2])
        print(frames_intersections.to_json())
        # Output: [
        #     {
        #         "index": 2,
        #         "figures": []
        #     }
        # ]

        # Note, two frames with the same index values are not equal
        frame_4 = sly.PointcloudEpisodeFrame(2)
        frames_intersections = pcd_episodes_fr_collection.intersection([frame_4])
        # Output:
        # ValueError: Different values for the same key 2

        # Difference, finds difference between collection and given list of PointcloudEpisodeFrame
        frames_difference = pcd_episodes_fr_collection.difference([frame_2])
        print(frames_difference.to_json())
        # Output: [
        #     {
        #         "index": 1,
        #         "figures": []
        #     }
        # ]

        # Merge, merges collection and given list of PointcloudEpisodeFrameCollection
        frame_3 = sly.PointcloudEpisodeFrame(3)
        frame_4 = sly.PointcloudEpisodeFrame(4)
        over_collection = sly.PointcloudEpisodeFrameCollection([frame_3, frame_4])
        merged_collection = pcd_episodes_fr_collection.merge(over_collection)
        print(merged_collection.to_json())
        # Output: [
        #     { "index": 3, "figures": [] },
        #     { "index": 4, "figures": [] },
        #     { "index": 1, "figures": [] },
        #     { "index": 2, "figures": [] }
        # ]

    """

    item_type = PointcloudEpisodeFrame

    def get(self, key: str, default: Optional[Any]=None) -> PointcloudEpisodeFrame:
        """
        Get a PointcloudEpisodeFrame by its key and set default value if it does not exist.

        :param key: Key of the PointcloudEpisodeFrame.
        :type key: str
        :param default: Default value to return if the key is not found (default: None).
        :type default: Optional[Any]
        :return: PointcloudEpisodeFrame object.
        :rtype: PointcloudEpisodeFrame
        :Usage example:

         .. code-block:: python

            frame_1 = sly.PointcloudEpisodeFrame(1)
            frame_2 = sly.PointcloudEpisodeFrame(2)


            pcd_episodes_fr_collection = sly.PointcloudEpisodeFrameCollection([frame_1, frame_2])
            print(pcd_episodes_fr_collection.get(2).to_json())
            # Output: {'index': 2, 'figures': []}
        """
    
        return super().get(key, default)

    def __iter__(self) -> Iterator[PointcloudEpisodeFrame]:
        return next(self)

    @classmethod
    def from_json(
        cls, 
        data: List[Dict], 
        objects: PointcloudObjectCollection, 
        frames_count: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudEpisodeFrameCollection:
        """
        Convert a list of json dicts to PointcloudEpisodeFrameCollection. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: List[dict]
        :param objects: PointcloudObjectCollection object.
        :type objects: PointcloudObjectCollection
        :param frames_count: Number of frames in point cloud episodes.
        :type frames_count: int, optional
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: FrameCollection object
        :rtype: :class:`FrameCollection`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            fr_collection_json = [
                {
                    "figures": [
                    {
                        "geometry": {
                        "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
                        "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
                        "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
                        },
                        "geometryType": "cuboid_3d",
                        "key": "c9fb727a9b53432fa0316d0a5b6043bc",
                        "objectKey": "0fc681681b1f4b12909ccf685c53b43e"
                    }
                    ],
                    "index": 7
                },
                {
                    "figures": [
                    {
                        "geometry": {
                        "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
                        "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
                        "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
                        },
                        "geometryType": "cuboid_3d",
                        "key": "c3df69c90fa14cf284906ebad1c360ae",
                        "objectKey": "0fc681681b1f4b12909ccf685c53b43e"
                    }
                    ],
                    "index": 10
                }
            ]

            objects = []
            fr_collection = sly.PointcloudEpisodeFrameCollection.from_json(fr_collection_json, objects)
        """

        return super().from_json(data, objects, frames_count=frames_count, key_id_map=key_id_map)

    @property
    def figures(self) -> List[PointcloudFigure]:
        """
        Get figures from all frames in collection.

        :return: List of figures from all frames in collection
        :rtype: :class:`List[PointcloudFigure]<supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
            from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

            # Create pointcloud object
            obj_class_car = sly.ObjClass('car', Cuboid3d)
            pointcloud_obj_car = sly.PointcloudObject(obj_class_car)
            objects = PointcloudObjectCollection([pointcloud_obj_car])

            # Create two figures
            frame_index_1 = 7
            position_1, rotation_1, dimension_1 = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
            cuboid_1 = Cuboid3d(position_1, rotation_1, dimension_1)

            frame_index_2 = 10
            position_2, rotation_2, dimension_2 = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
            cuboid_2 = Cuboid3d(position_2, rotation_2, dimension_2)

            figure_1 = sly.PointcloudFigure(pointcloud_obj_car, cuboid_1, frame_index=frame_index_1)
            figure_1 = sly.PointcloudFigure(pointcloud_obj_car, cuboid_2, frame_index=frame_index_2)

            # Create two frames for collection
            frame_1 = sly.PointcloudEpisodeFrame(frame_index_1, figures=[figure_1])
            frame_2 = sly.PointcloudEpisodeFrame(frame_index_2, figures=[figure_2])

            # Create PointcloudEpisodeFrameCollection
            pcd_episodes_fr_collection = sly.PointcloudEpisodeFrameCollection([frame_1, frame_2])

            print([figure.to_json() for figure in pcd_episodes_fr_collection.figures])
            # Output: [
            #     {
            #         "geometry": {
            #         "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
            #         "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
            #         "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
            #         },
            #         "geometryType": "cuboid_3d",
            #         "key": "e35386df7c994485ba7186e1a8c67361",
            #         "objectKey": "c43ac88f2cb74852817390ca3b749ffd"
            #     },
            #     {
            #         "geometry": {
            #         "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
            #         "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
            #         "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
            #         },
            #         "geometryType": "cuboid_3d",
            #         "key": "1b09a434c20f47faab3f0d7efdb4cbbf",
            #         "objectKey": "c43ac88f2cb74852817390ca3b749ffd"
            #     }
            # ]
        """

        return super().figures