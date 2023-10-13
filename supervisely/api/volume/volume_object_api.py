# coding: utf-8

from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_object_collection import (
    VolumeObjectCollection,
)
from supervisely.api.entity_annotation.object_api import ObjectApi


class VolumeObjectApi(ObjectApi):
    """
    :class:`VolumeObject<supervisely.volume_annotation.volume_object.VolumeObject>` for :class:`VolumeAnnotation<supervisely.volume_annotation.volume_annotation.VolumeAnnotation>`.
    """

    def append_bulk(
        self, volume_id: int, objects: VolumeObjectCollection, key_id_map: KeyIdMap = None
    ):
        """
        Add Tags to Annotation Objects

        :param volume_id: Volume ID in Supervidely.
        :type volume_id: int
        :param objects: VolumeAnnotation objects.
        :type objects: VolumeObjectCollection
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: List of objects IDs
        :rtype: :class:`List[int]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.volume_annotation.volume_tag import VolumeTag
            from supervisely.video_annotation.key_id_map import KeyIdMap

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 17209
            volume_id = 19402023

            meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(meta_json)

            key_id_map = KeyIdMap()
            ann_info = api.volume.annotation.download(volume_id)
            ann = sly.VolumeAnnotation.from_json(ann_info, project_meta, key_id_map)

            api.volume.object.append_bulk(volume_id, ann.objects, key_id_map)
        """

        info = self._api.volume.get_info_by_id(volume_id)
        return self._append_bulk(
            self._api.volume.tag,
            volume_id,
            info.project_id,
            info.dataset_id,
            objects,
            key_id_map,
        )
