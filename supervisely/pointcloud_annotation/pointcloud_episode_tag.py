from __future__ import annotations
import uuid
from typing import Optional, Dict, Union, Tuple
from supervisely.annotation.tag_meta import TagMeta
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_tag import VideoTag


class PointcloudEpisodeTag(VideoTag):
    """
    PointcloudEpisodeTag object for :class:`PointcloudEpisodeAnnotation<supervisely.pointcloud_annotation.pointcloud_episode_annotation.PointcloudEpisodeAnnotation>`. :class:`PointcloudEpisodeTag<PointcloudEpisodeTag>` object is immutable.

    :param meta: General information about point cloud episodes Tag.
    :type meta: TagMeta
    :param value: point cloud episodes Tag value. Depends on :class:`TagValueType<TagValueType>` of :class:`TagMeta<TagMeta>`.
    :type value: Optional[Union[str, int, float]]
    :param frame_range: point cloud episodes Tag frame range.
    :type frame_range: Tuple[int, int] or List[int, int], optional
    :param key: uuid.UUID object.
    :type key: uuid.UUID, optional
    :param sly_id: PointcloudEpisodeTag ID in Supervisely.
    :type sly_id: int, optional
    :param labeler_login: Login of user who created PointcloudEpisodeTag.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when PointcloudEpisodeTag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when PointcloudEpisodeTag was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        meta_car = sly.TagMeta('car', sly.TagValueType.NONE)
        # Now we can create a VideoTag using our TagMeta
        tag_car = sly.PointcloudEpisodeTag(meta_car)
        # When you are creating a new Tag
        # Tag.value is automatically cross-checked against your TagMeta value type to make sure the value is valid.
        # If we now try to add a value to our newly created Tag, we receive "ValueError", because our TagMeta value type is "NONE"
        tag_car = sly.PointcloudEpisodeTag(meta_car, value="Bus")
        # Output: ValueError: Tag car can not have value Bus

        # Let's create another Tag with a string value type and frame range
        meta_car = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
        tag_car = sly.PointcloudEpisodeTag(meta_car, value="red", frame_range=(5, 10))

        # Now let's create a Tag using TagMeta with "ONEOF_STRING" value type
        # In order to use "oneof_string value type", you must initialize a variable with possible values(see class TagMeta for more information)
        colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
        meta_car_color = sly.TagMeta('car color', sly.TagValueType.ONEOF_STRING, possible_values=colors)
        tag_car_color = sly.PointcloudEpisodeTag(meta_car_color, value="white", frame_range=(15, 20))

        # If given value is not in a list of possible Tags, ValueError will be raised
        tag_car_color = sly.PointcloudEpisodeTag(meta_car_color, value="yellow")
        # Output: ValueError: Tag car color can not have value yellow
    """

    @classmethod
    def from_json(
        cls,
        data: Dict,
        tag_meta_collection: TagMetaCollection,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> PointcloudEpisodeTag:
        """
        Convert a json dict to PointcloudEpisodeTag. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: PointcloudEpisodeTag in json format as a dict.
        :type data: dict
        :param tag_meta_collection: :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` object.
        :type tag_meta_collection: TagMetaCollection
        :param key_id_map: Key ID Map object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudEpisodeTag object
        :rtype: :class:`PointcloudEpisodeTag<PointcloudEpisodeTag>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            tag_car_color_json = {
                "frameRange": [15, 20],
                "key": "da9ca75e97744fc5aaf24d6be2eb2832",
                "name": "car color",
                "value": "white"
            }

            colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
            meta_car_color = sly.TagMeta('car color', sly.TagValueType.ONEOF_STRING, possible_values=colors)
            meta_car_collection = sly.TagMetaCollection([meta_car_color])

            tag_car_color = sly.PointcloudEpisodeTag.from_json(tag_car_color_json, meta_car_collection)
        """

        return super().from_json(data, tag_meta_collection, key_id_map=key_id_map)

    def __eq__(self, other: PointcloudEpisodeTag) -> bool:
        return super().__eq__(other)

    def clone(
        self,
        meta: Optional[TagMeta] = None,
        value: Optional[Union[str, int, float]] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        key: Optional[uuid.UUID] = None,
        sly_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> PointcloudEpisodeTag:
        """
        Makes a copy of PointcloudEpisodeTag with new fields, if fields are given, otherwise it will use fields of the original PointcloudEpisodeTag.

        :param meta: General information about PointcloudEpisodeTag.
        :type meta: TagMeta, optional
        :param value: PointcloudEpisodeTag value. Depends on :class:`TagValueType<TagValueType>` of :class:`TagMeta<TagMeta>`.
        :type value: Optional[Union[str, int, float]]
        :param frame_range: PointcloudEpisodeTag frame range.
        :type frame_range: Optional[Union[Tuple[int, int], List[int, int]]]
        :param key: uuid.UUID object.
        :type key: uuid.UUID, optional
        :param sly_id: PointcloudEpisodeTag ID in Supervisely.
        :type sly_id: int, optional
        :param labeler_login: Login of user who created PointcloudEpisodeTag.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when PointcloudEpisodeTag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when PointcloudEpisodeTag was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
            meta_car_color = sly.TagMeta('car color', sly.TagValueType.ONEOF_STRING, possible_values=colors)

            tag_car_color = sly.PointcloudEpisodeTag(meta_car_color, value="white", frame_range=(15, 20))


            meta_bus = sly.TagMeta('bus', sly.TagValueType.ANY_STRING)

            new_tag = tag_car_color.clone(meta=meta_bus, frame_range=(15, 30), key=tag_car_color.key())
            print(new_tag.to_json())
            # Output: {
            #     "frameRange": [15, 30],
            #     "key": "4360b25778144141aa4f1a0d775a0a7a",
            #     "name": "bus",
            #     "value": "white"
            # }
        """

        return super().clone(
            meta=meta,
            value=value,
            frame_range=frame_range,
            key=key,
            sly_id=sly_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
