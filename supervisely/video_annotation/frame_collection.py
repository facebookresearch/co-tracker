# coding: utf-8

# docs
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.video_annotation.video_figure import VideoFigure

from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.video_annotation.frame import Frame
from supervisely.api.module_api import ApiField


class FrameCollection(KeyIndexedCollection):
    """
    Collection with :class:`Frame<supervisely.video_annotation.frame.Frame>` instances. :class:`FrameCollection<FrameCollection>` object is immutable.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create two frames for collection
        fr_index_1 = 7
        frame_1 = sly.Frame(fr_index_1)
        fr_index_2 = 10
        frame_2 = sly.Frame(fr_index_2)

        # Create FrameCollection
        fr_collection = sly.FrameCollection([frame_1, frame_2])
        print(fr_collection.to_json())
        # Output: [
        #     {
        #         "index": 7,
        #         "figures": []
        #     },
        #     {
        #         "index": 10,
        #         "figures": []
        #     }
        # ]

        # Add item to FrameCollection
        frame_3 = sly.Frame(12)
        # Remember that TagCollection is immutable, and we need to assign new instance of TagCollection to a new variable
        new_fr_collection = fr_collection.add(frame_3)
        print(new_fr_collection.to_json())
        # Output: [
        #     {
        #         "index": 7,
        #         "figures": []
        #     },
        #     {
        #         "index": 10,
        #         "figures": []
        #     },
        #     {
        #         "index": 12,
        #         "figures": []
        #     }
        # ]

        # You can also add multiple items to collection
        frame_3 = sly.Frame(12)
        frame_4 = sly.Frame(15)
        # Remember that TagCollection is immutable, and we need to assign new instance of TagCollection to a new variable
        new_fr_collection = fr_collection.add_items([frame_3, frame_4])
        print(new_fr_collection.to_json())
        # Output: [
        #     {
        #         "index": 7,
        #         "figures": []
        #     },
        #     {
        #         "index": 10,
        #         "figures": []
        #     },
        #     {
        #         "index": 12,
        #         "figures": []
        #     },
        #     {
        #         "index": 15,
        #         "figures": []
        #     }
        # ]

        # Has key, checks if given key exist in collection
        fr_collection.has_key(7)
        # Output: True

        # Intersection, finds intersection of given list of instances with collection items
        frame_1 = sly.Frame(7)
        frame_2 = sly.Frame(10)
        fr_collection = sly.FrameCollection([frame_1, frame_2])
        frame_3 = sly.Frame(12)
        frames_intersections = fr_collection.intersection([frame_3])
        print(frames_intersections.to_json())
        # Output: []

        frames_intersections = fr_collection.intersection([frame_2])
        print(frames_intersections.to_json())
        # Output: [
        #     {
        #         "index": 10,
        #         "figures": []
        #     }
        # ]

        # Note, two frames with the same index values are not equal
        frame_4 = sly.Frame(10)
        frames_intersections = fr_collection.intersection([frame_4])
        # Output:
        # ValueError: Different values for the same key 10

        # Difference, finds difference between collection and given list of Frames
        frames_difference = fr_collection.difference([frame_2])
        print(frames_difference.to_json())
        # Output: [
        #     {
        #         "index": 7,
        #         "figures": []
        #     }
        # ]

        # Merge, merges collection and given list of FrameCollection
        frame_3 = sly.Frame(12)
        frame_4 = sly.Frame(15)
        over_collection = sly.FrameCollection([frame_3, frame_4])
        merged_collection = fr_collection.merge(over_collection)
        print(merged_collection.to_json())
        # Output: [
        #     {
        #         "index": 12,
        #         "figures": []
        #     },
        #     {
        #         "index": 15,
        #         "figures": []
        #     },
        #     {
        #         "index": 7,
        #         "figures": []
        #     },
        #     {
        #         "index": 10,
        #         "figures": []
        #     }
        # ]
    """
    item_type = Frame

    def get(self, key: str, default: Optional[Any]=None) -> Frame:
        """
        Get item from collection with given key(name) and set a default if item does not exist.

        :param key: Name of Frame in collection.
        :type key: str
        :param default: The value that is returned if there is no key in the collection.
        :type default:  Optional[Any]
        :return: :class:`Frame<supervisely.video_annotation.frame.Frame>`, :class:`Slice<supervisely.volume_annotation.slice.Slice>` or :class:`PointcloudEpisodeFrame<supervisely.pointcloud_annotation.pointcloud_episode_frame.PointcloudEpisodeFrame>` object
        :rtype: :class:`KeyObject<KeyObject>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            frame_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            class_car = sly.ObjClass('car', sly.Rectangle)
            object_car = sly.VideoObject(class_car)
            figure_car = sly.VideoFigure(object_car, geometry, frame_index)

            frame = sly.Frame(frame_index, figures=[figure_car])
            frame_collection = sly.FrameCollection([frame])

            item = frame_collection.get(frame_index)
            pprint(item.to_json())
            # Output: {
            #     "figures": [
            #         {
            #         "geometry": {
            #             "points": {
            #             "exterior": [
            #                 [0, 0],
            #                 [100, 100]
            #             ],
            #             "interior": []
            #             }
            #         },
            #         "geometryType": "rectangle",
            #         "key": "713968a7d5384709bc5d4e63cd4535f2",
            #         "objectKey": "3342e68eff3b44dcb75712499265be55"
            #         }
            #     ],
            #     "index": 7
            # }
        """
        return super().get(key, default)

    def to_json(self, key_id_map: KeyIdMap=None) -> List[Dict]:
        """
        Convert the FrameCollection to a list of json dicts. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: List of dicts in json format
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            frame_1 = sly.Frame(7)
            frame_2 = sly.Frame(10)

            # Create FrameCollection
            fr_collection = sly.FrameCollection([frame_1, frame_2])
            print(fr_collection.to_json())
            # Output: [
            #     {
            #         "index": 7,
            #         "figures": []
            #     },
            #     {
            #         "index": 10,
            #         "figures": []
            #     }
            # ]
        """
        return [frame.to_json(key_id_map) for frame in self]

    @classmethod
    def from_json(cls, data: List[Dict], objects: VideoObjectCollection, frames_count: Optional[int] = None,
                  key_id_map: Optional[KeyIdMap] = None) -> FrameCollection:
        """
        Convert a list of json dicts to FrameCollection. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: List[dict]
        :param objects: VideoObjectCollection object.
        :type objects: VideoObjectCollection
        :param frames_count: Number of frames in video.
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
                    "index": 7,
                    "figures": []
                },
                {
                    "index": 10,
                    "figures": []
                }
            ]

            objects = []
            fr_collection = sly.FrameCollection.from_json(fr_collection_json, objects)
        """
        frames = [cls.item_type.from_json(frame_json, objects, frames_count, key_id_map) for frame_json in data]
        return cls(frames)

    def __str__(self):
        return "Frames:\n" + super(FrameCollection, self).__str__()

    @property
    def figures(self) -> List[VideoFigure]:
        """
        Get figures from all frames in collection.

        :return: List of figures from all frames in collection
        :rtype: :class:`List[VideoFigure]<supervisely.video_annotation.video_figure.VideoFigure>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            fr_index_1 = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_object_car = sly.VideoObject(obj_class_car)
            video_figure_car = sly.VideoFigure(video_object_car, geometry, fr_index_1)
            frame_1 = sly.Frame(fr_index_1, figures=[video_figure_car])

            fr_index_2 = 10
            geometry = sly.Rectangle(0, 0, 500, 600)
            obj_class_bus = sly.ObjClass('bus', sly.Rectangle)
            video_object_bus = sly.VideoObject(obj_class_bus)
            video_figure_bus = sly.VideoFigure(video_object_bus, geometry, fr_index_2)
            frame_2 = sly.Frame(fr_index_2, figures=[video_figure_bus])

            fr_collection = sly.FrameCollection([frame_1, frame_2])
            figures = fr_collection.figures
        """
        figures_array = []
        for frame in self:
            figures_array.extend(frame.figures)
        return figures_array

    def get_figures_and_keys(self, key_id_map: KeyIdMap) -> Tuple[List[Dict], List[str]]:
        """
        Get figures from all frames in collection in json format, keys from all figures in frames in collection.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :return: Figures from all frames in collection in json format, keys from all figures in frames in collection
        :rtype: :class:`Tuple[list, list]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.key_id_map import KeyIdMap

            key_id_map = KeyIdMap()

            fr_index_1 = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_object_car = sly.VideoObject(obj_class_car)
            video_figure_car = sly.VideoFigure(video_object_car, geometry, fr_index_1)
            frame_1 = sly.Frame(fr_index_1, figures=[video_figure_car])

            fr_index_2 = 10
            geometry = sly.Rectangle(0, 0, 500, 600)
            obj_class_bus = sly.ObjClass('bus', sly.Rectangle)
            video_object_bus = sly.VideoObject(obj_class_bus)
            video_figure_bus = sly.VideoFigure(video_object_bus, geometry, fr_index_2)
            frame_2 = sly.Frame(fr_index_2, figures=[video_figure_bus])

            fr_collection = sly.FrameCollection([frame_1, frame_2])
            figures, keys = fr_collection.get_figures_and_keys(key_id_map)
            print(keys) # [UUID('0ac041b2-314e-4f6b-9d38-704b341fb383'), UUID('88aa1cb3-b1e3-480f-8ace-6346c9a9daba')]

            print(figures)
            # Output: [
            #     {
            #         "key": "a8cae05d6b8c4a67b18004130941fdec",
            #         "objectKey": "cc9a9475d360481c9753f8ac3c63f8b7",
            #         "geometryType": "rectangle",
            #         "geometry": {
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         0,
            #                         0
            #                     ],
            #                     [
            #                         100,
            #                         100
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         },
            #         "meta": {
            #             "frame": 7
            #         }
            #     },
            #     {
            #         "key": "6e00287acc4644dfb21d67406534080b",
            #         "objectKey": "cad78d53ffc84e69a28f5f8941be9021",
            #         "geometryType": "rectangle",
            #         "geometry": {
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         0,
            #                         0
            #                     ],
            #                     [
            #                         600,
            #                         500
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         },
            #         "meta": {
            #             "frame": 10
            #         }
            #     }
            # ]
        """
        keys = []
        figures_json = []
        for frame in self:
            for figure in frame.figures:
                keys.append(figure.key())
                figure_json = figure.to_json(key_id_map)
                figure_json[ApiField.META] = figure.get_meta()
                figures_json.append(figure_json)
        return figures_json, keys
