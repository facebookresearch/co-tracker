# coding: utf-8

# docs
from __future__ import annotations
from typing import List, Optional
from supervisely.video_annotation.video_figure import VideoFigure

from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.api.entity_annotation.figure_api import FigureApi


class VideoFigureApi(FigureApi):
    """
    :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` for a single video.
    """

    def create(
        self,
        video_id: int,
        object_id: int,
        frame_index: int,
        geometry_json: dict,
        geometry_type: str,
        track_id: Optional[int] = None,
    ) -> int:
        """
        Create new VideoFigure for given frame in given video ID.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param object_id: ID of the object to which the VideoFigure belongs.
        :type object_id: int
        :param frame_index: Number of the frame to add VideoFigure.
        :type frame_index: int
        :param geometry_json: Parameters of geometry for VideoFigure.
        :type geometry_json: dict
        :param geometry_type: Type of VideoFigure geometry.
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

            video_id = 198703211
            object_id = 152118
            frame_idx = 0
            geometry_json = {'points': {'exterior': [[500, 500], [1555, 1500]], 'interior': []}}
            geometry_type = 'rectangle'

            figure_id = api.video.figure.create(video_id, object_id, frame_idx, geometry_json, geometry_type) # 643182610
        """

        return super().create(
            video_id,
            object_id,
            {ApiField.FRAME: frame_index},
            geometry_json,
            geometry_type,
            track_id,
        )

    def append_bulk(self, video_id: int, figures: List[VideoFigure], key_id_map: KeyIdMap) -> None:
        """
        Add VideoFigures to given Video by ID.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param figures: List of VideoFigures to append.
        :type figures: List[VideoFigure]
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

            video_id = 198703212
            ann_info = api.video.annotation.download(video_id)
            ann = sly.VideoAnnotation.from_json(ann_info, meta, key_id_map)
            figures = ann.figures[:5]
            api.video.figure.append_bulk(video_id, figures, key_id_map)
        """

        keys = []
        figures_json = []
        for figure in figures:
            keys.append(figure.key())
            figures_json.append(figure.to_json(key_id_map, save_meta=True))

        self._append_bulk(video_id, figures_json, keys, key_id_map)
