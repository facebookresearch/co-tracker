# coding: utf-8

import uuid
from supervisely.annotation.tag import Tag, TagJsonFields
from supervisely._utils import take_with_default
from supervisely.volume_annotation.constants import KEY, ID
from supervisely.video_annotation.key_id_map import KeyIdMap


class VolumeTag(Tag):
    """
    VolumeTag object for :class:`VolumeAnnotation<supervisely.volume_annotation.volume_annotation.VolumeAnnotation>`. :class:`VolumeTag<VolumeTag>` object is immutable.

    :param meta: General information about Volume Tag.
    :type meta: TagMeta
    :param value: Volume Tag value. Depends on :class:`TagValueType<TagValueType>` of :class:`TagMeta<TagMeta>`.
    :type value: Optional[Union[str, int, float]]
    :param key: uuid.UUID object.
    :type key: uuid.UUID, optional
    :param sly_id: Volume Tag ID in Supervisely.
    :type sly_id: int, optional
    :param labeler_login: Login of user who created VolumeTag.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when VolumeTag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when VolumeTag was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.volume_annotation.volume_tag import VolumeTag

        meta_brain = sly.TagMeta('brain', sly.TagValueType.NONE)
        # Now we can create a VolumeTag using our TagMeta
        tag_brain = VolumeTag(meta_brain)
        # When you are creating a new Tag
        # Tag.value is automatically cross-checked against your TagMeta value type to make sure the value is valid.
        # If we now try to add a value to our newly created Tag, we receive "ValueError", because our TagMeta value type is "NONE"
        tag_dog = VolumeTag(meta_brain, value="Brain")
        # Output: ValueError: Tag brain can not have value Husky

        # Let's create another Tag with a string value type
        meta_heart = sly.TagMeta('heart', sly.TagValueType.ANY_STRING)
        tag_heart = VolumeTag(meta_heart, value="Heart")

        # Now let's create a Tag using TagMeta with "ONEOF_STRING" value type
        # In order to use "oneof_string value type", you must initialize a variable with possible values(see class TagMeta for more information)
        colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
        meta_lang_color = sly.TagMeta('lang color', sly.TagValueType.ONEOF_STRING, possible_values=colors)
        tag_lang_color = VolumeTag(meta_lang_color, value="white")

        # If given value is not in a list of possible Tags, ValueError will be raised
        tag_lang_color = VolumeTag(meta_lang_color, value="yellow")
        # Output: ValueError: Tag lang color can not have value yellow
    """

    def __init__(
        self,
        meta,
        value=None,
        key=None,
        sly_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        super(VolumeTag, self).__init__(
            meta,
            value=value,
            sly_id=sly_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        self._key = take_with_default(key, uuid.uuid4())

    def key(self) -> str:
        """
        Get key value.

        :return: Get key value.
        :rtype: str
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.volume_annotation.volume_tag import VolumeTag

            meta_heart = sly.TagMeta('heart', sly.TagValueType.NONE)
            tag_heart = VolumeTag(meta_heart)

            print(tag_heart.key()) 
            # Output: 2d9cd2cd-f89c-40a9-8675-1b0773fa250d
        """
        return self._key

    def to_json(self, key_id_map: KeyIdMap = None):
        """
        Convert the VolumeTag to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: Key ID Map object.
        :type key_id_map: KeyIdMap, optional
        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.volume_annotation.volume_tag import VolumeTag
            meta_heart = sly.TagMeta('heart', sly.TagValueType.NONE)
            tag_heart = VolumeTag(meta_heart)

            tag_heart_json = tag_heart.to_json()

            print(tag_heart_json)
            # Output: {
            #     "name": "heart",
            #     "key": "058ad7993a534082b4d94cc52542a97d"
            # }
        """

        data_json = super(VolumeTag, self).to_json()
        if type(data_json) is str:
            # @TODO: case when tag has no value, super.to_json() returns tag name
            data_json = {TagJsonFields.TAG_NAME: data_json}
        data_json[KEY] = self.key().hex

        if key_id_map is not None:
            item_id = key_id_map.get_tag_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

        return data_json

    @classmethod
    def from_json(cls, data: dict, tag_meta_collection, key_id_map: KeyIdMap = None):
        """
        Convert a json dict to VolumeTag. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: VolumeTag in json format as a dict.
        :type data: dict
        :param tag_meta_collection: :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` object.
        :type tag_meta_collection: TagMetaCollection
        :param key_id_map: Key ID Map object.
        :type key_id_map: KeyIdMap, optional
        :return: VolumeTag object
        :rtype: :class:`VolumeTag<VolumeTag>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            tag_heart_json = {
                "name": "heart",
                "value": "777"
            }

            from supervisely.volumme_annotation.volume_tag import VolumeTag
            meta_heart = sly.TagMeta('heart', sly.TagValueType.ANY_STRING)
            meta_collection = sly.TagMetaCollection([meta_heart])

            tag_heart = VolumeTag.from_json(tag_heart_json, meta_collection)
        """

        temp = super(VolumeTag, cls).from_json(data, tag_meta_collection)
        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_tag(key, data.get(ID, None))

        return cls(
            meta=temp.meta,
            value=temp.value,
            key=key,
            sly_id=temp.sly_id,
            labeler_login=temp.labeler_login,
            updated_at=temp.updated_at,
            created_at=temp.created_at,
        )

    def clone(
        self,
        meta=None,
        value=None,
        key=None,
        sly_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        """
        Makes a copy of VolumeTag with new fields, if fields are given, otherwise it will use fields of the original VolumeTag.

        :param meta: General information about VolumeTag.
        :type meta: VolumeTag, optional
        :param value: VolumeTag value. Depends on :class:`TagValueType<TagValueType>` of :class:`TagMeta<TagMeta>`.
        :type value: str or int or float or None, optional
        :param key: uuid.UUID object.
        :type key: uuid.UUID, optional
        :param sly_id: VolumeTag ID in Supervisely.
        :type sly_id: int, optional
        :param labeler_login: Login of user who created VolumeTag.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when VolumeTag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when VolumeTag was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.volume_annotation.volume_tag import VolumeTag

            meta_heart = sly.TagMeta('heart_tag', sly.TagValueType.ANY_STRING)
            heart_tag = VolumeTag(meta_heart, value='Heart')

            meta_heart_2 = sly.TagMeta('heart tag 2', sly.TagValueType.ANY_STRING)

            new_tag = car_tag.clone(meta=meta_heart_2, key=car_tag.key())
            new_tag_json = new_tag.to_json()

            print(new_tag_json)
            # Output: {
            #     "name": "heart tag 2",
            #     "value": "Heart",
            #     "key": "360438485fd34264921ca19bd43b0b71"
            # }
        """

        return VolumeTag(
            meta=take_with_default(meta, self.meta),
            value=take_with_default(value, self.value),
            key=take_with_default(key, self.key),
            sly_id=take_with_default(sly_id, self.sly_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )
