# coding: utf-8
from typing import Union
from supervisely.api.module_api import ApiField
from supervisely.api.entity_annotation.tag_api import TagApi


class VolumeTagApi(TagApi):
    """
    :class:`VolumeTag<supervisely.volume_annotation.volume_tag.VolumeTag>` for a single volume. :class:`VolumeTagApi<VolumeTagApi>` object is immutable.
    """

    _entity_id_field = ApiField.ENTITY_ID
    _method_bulk_add = "volumes.tags.bulk.add"

    def remove_from_volume(self, tag_id: int):
        """
        Remove tag from volume.

        :param tag_id: VolumeTag ID in Supervisely.
        :type tag_id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.volume.tag.remove_from_volume(volume_tag_id)

        """

        self._api.post("volumes.tags.remove", {ApiField.ID: tag_id})

    def update_value(self, tag_id: int, tag_value: Union[str, int]):
        """
        Update VolumeTag value.

        :param tag_id: VolumeTag ID in Supervisely.
        :type tag_id: int
        :param tag_value: New VolumeTag value.
        :type tag_value: str or int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.volume.tag.update_value(volume_tag_id, 'new_tag_value')
        """

        self._api.post(
            "volumes.tags.update-value",
            {ApiField.ID: tag_id, ApiField.VALUE: tag_value},
        )
