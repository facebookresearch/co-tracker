# coding: utf-8

# docs
from __future__ import annotations
from typing import Tuple, Dict, Optional, List
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_object_collection import VideoObjectCollection

from supervisely._utils import take_with_default
from supervisely.video_annotation.constants import FIGURES, INDEX
from supervisely.video_annotation.video_figure import VideoFigure
from supervisely.collection.key_indexed_collection import KeyObject


class Frame(KeyObject):
    """
    Frame object for :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`. :class:`Frame<Frame>` object is immutable.

    :param index: Index of the Frame.
    :type index: int
    :param figures: List of :class:`VideoFigures<supervisely.video_annotation.video_figure.VideoFigure>`.
    :type figures: list, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        frame_index = 7
        geometry = sly.Rectangle(0, 0, 100, 100)
        class_car = sly.ObjClass('car', sly.Rectangle)
        object_car = sly.VideoObject(class_car)
        figure_car = sly.VideoFigure(object_car, geometry, frame_index)

        frame = sly.Frame(frame_index, figures=[figure_car])
        print(frame.to_json())
        # Output: {
        #     "index": 7,
        #     "figures": [
        #         {
        #             "key": "39f3eb15791f4c72b7cdb98c17b3f0f1",
        #             "objectKey": "319814af474941a98ca208c3fad5ed81",
        #             "geometryType": "rectangle",
        #             "geometry": {
        #                 "points": {
        #                     "exterior": [
        #                         [
        #                             0,
        #                             0
        #                         ],
        #                         [
        #                             100,
        #                             100
        #                         ]
        #                     ],
        #                     "interior": []
        #                 }
        #             }
        #         }
        #     ]
        # }
    """
    figure_type = VideoFigure

    def __init__(self, index: int, figures: Optional[List[VideoFigure]]=None):
        self._index = index
        self._figures = take_with_default(figures, [])

    @property
    def index(self) -> int:
        """
        Frame index.

        :return: Frame index.
        :rtype: int
        :Usage example:

         .. code-block:: python

            frame_index = frame.index # 7
        """
        return self._index

    def key(self) -> int:
        """
        Get Frame key (index) value.

        :return: Frame key (index) value
        :rtype: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            frame_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            class_car = sly.ObjClass('car', sly.Rectangle)
            object_car = sly.VideoObject(class_car)
            figure_car = sly.VideoFigure(object_car, geometry, frame_index)

            frame = sly.Frame(frame_index, figures=[figure_car])
            print(frame.key())
            # Output: 7
        """

        return self._index

    @property
    def figures(self) -> List[VideoFigure]:
        """
        Frame figures.

        :return: List of figures on Frame.
        :rtype: :class:`List[VideoFigure]<supervisely.video_annotation.video_figure.VideoFigure>`
        :Usage example:

         .. code-block:: python

            frame_figures = frame.figures
        """
        return self._figures.copy()

    def validate_figures_bounds(self, img_size: Optional[Tuple[int, int]]=None) -> None:
        """
        Checks if image with given size contains a figure.

        :param img_size: Size of the image (height, width).
        :type img_size: Tuple[int, int], optional
        :raises: :class:`OutOfImageBoundsException<supervisely.video_annotation.video_figure.OutOfImageBoundsException>`, if figure is out of image bounds
        :return: None
        :rtype: :class:`NoneType`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            frame_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            class_car = sly.ObjClass('car', sly.Rectangle)
            object_car = sly.VideoObject(class_car)
            figure_car = sly.VideoFigure(object_car, geometry, frame_index)
            frame = sly.Frame(frame_index, figures=[figure_car])

            image_size = (20, 200)
            frame.validate_figures_bounds(image_size)
            # raise OutOfImageBoundsException("Figure is out of image bounds")
        """
        if img_size is None:
            return
        for figure in self._figures:
            figure.validate_bounds(img_size, _auto_correct=False)

    def to_json(self, key_id_map: KeyIdMap = None) -> Dict:
        """
        Convert the Frame to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: Json format as a dict
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            frame_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            class_car = sly.ObjClass('car', sly.Rectangle)
            object_car = sly.VideoObject(class_car)
            figure_car = sly.VideoFigure(object_car, geometry, frame_index)

            frame = sly.Frame(frame_index, figures=[figure_car])
            frame_json = frame.to_json()
            print(frame_json)
            # Output: {
            #     "index": 7,
            #     "figures": [
            #         {
            #             "key": "39f3eb15791f4c72b7cdb98c17b3f0f1",
            #             "objectKey": "319814af474941a98ca208c3fad5ed81",
            #             "geometryType": "rectangle",
            #             "geometry": {
            #                 "points": {
            #                     "exterior": [
            #                         [
            #                             0,
            #                             0
            #                         ],
            #                         [
            #                             100,
            #                             100
            #                         ]
            #                     ],
            #                     "interior": []
            #                 }
            #             }
            #         }
            #     ]
            # }

        """
        data_json = {
            INDEX: self.index,
            FIGURES: [figure.to_json(key_id_map) for figure in self.figures],
        }
        return data_json

    @classmethod
    def from_json(cls, data: Dict, objects: VideoObjectCollection, frames_count: Optional[int]=None, key_id_map: Optional[KeyIdMap]=None) -> Frame:
        """
        Convert a json dict to Frame. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Dict in json format.
        :type data: dict
        :param objects: VideoObjectCollection object.
        :type objects: VideoObjectCollection
        :param frames_count: Number of frames in video.
        :type frames_count: int, optional
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :raises: :class:`ValueError` if frame index < 0 and if frame index > number of frames in video
        :return: Frame object
        :rtype: :class:`Frame`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            frame_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            class_car = sly.ObjClass('car', sly.Rectangle)
            object_car = sly.VideoObject(class_car)
            figure_car = sly.VideoFigure(object_car, geometry, frame_index)

            frame = sly.Frame(frame_index, figures=[figure_car])
            frame_json = frame.to_json()

            video_obj_coll = sly.VideoObjectCollection([object_car])
            frame_car = sly.Frame.from_json(frame_json, video_obj_coll)
        """
        index = data[INDEX]
        if index < 0:
            raise ValueError("Frame Index have to be >= 0")

        if frames_count is not None:
            if index > frames_count:
                raise ValueError(
                    "Item contains {} frames. Frame index is {}".format(
                        frames_count, index
                    )
                )

        figures = []
        for figure_json in data.get(FIGURES, []):
            figure = cls.figure_type.from_json(figure_json, objects, index, key_id_map)
            figures.append(figure)
        return cls(index=index, figures=figures)

    def clone(self, index: Optional[int] = None, figures: Optional[List[VideoFigure]] = None) -> Frame:
        """
        Makes a copy of Frame with new fields, if fields are given, otherwise it will use fields of the original Frame.

        :param index: Index of the Frame.
        :type index: int, optional
        :param figures: List of :class:`VideoFigures<supervisely.video_annotation.video_figure.VideoFigure>`.
        :type figures: list, optional
        :return: Frame object
        :rtype: :class:`Frame`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            frame_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            class_car = sly.ObjClass('car', sly.Rectangle)
            object_car = sly.VideoObject(class_car)
            figure_car = sly.VideoFigure(object_car, geometry, frame_index)
            frame = sly.Frame(frame_index, figures=[figure_car])

            # Remember that Frame object is immutable, and we need to assign new instance of Frame to a new variable
            new_frame = frame.clone(index=100, figures=[])
            print(new_frame.to_json())
            # Output: {
            #     "index": 100,
            #     "figures": []
            # }
        """
        return self.__class__(index=take_with_default(index, self.index),
                              figures=take_with_default(figures, self.figures))
