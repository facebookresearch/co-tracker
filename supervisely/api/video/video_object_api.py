# coding: utf-8

# docs
from typing import List, Optional

from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.api.entity_annotation.object_api import ObjectApi
from supervisely.api.video.video_tag_api import VideoObjectTagApi


class VideoObjectApi(ObjectApi):
    """
    :class:`VideoObject<supervisely.video_annotation.video_object.VideoObject>` for :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`.
    """

    def __init__(self, api):
        super().__init__(api)
        self.tag = VideoObjectTagApi(api)

    def append_bulk(
        self,
        video_id: int,
        objects: VideoObjectCollection,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> List[int]:
        """
        Add Objects to Annotation Objects.

        :param video_id: Video ID in Supervidely.
        :type video_id: int
        :param objects: VideoAnnotation objects.
        :type objects: VideoObjectCollection
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: List of objects IDs
        :rtype: :class:`List[int]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.video_annotation.key_id_map import KeyIdMap

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 17209
            video_id = 19402023

            meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(meta_json)

            key_id_map = KeyIdMap()
            ann_info = api.video.annotation.download(video_id)
            ann = sly.VideoAnnotation.from_json(ann_info, project_meta, key_id_map)

            api.video.object.append_bulk(video_id, ann.objects, key_id_map)
        """

        info = self._api.video.get_info_by_id(video_id)
        return self._append_bulk(
            self._api.video.tag, video_id, info.project_id, info.dataset_id, objects, key_id_map
        )
