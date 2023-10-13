from __future__ import annotations
from typing import Optional, List, Dict, Any, Iterator

from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.annotation.tag_collection import TagCollection
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.pointcloud_annotation.pointcloud_tag import PointcloudTag
from supervisely.annotation.tag_meta import TagMeta
import supervisely.sly_logger as logger


class PointcloudTagCollection(TagCollection):
    """
    Collection with :class:`PointcloudTag<supervisely.pointcloud_annotation.pointcloud_tag.PointcloudTag>` instances.
    :class:`PointcloudTagCollection<PointcloudTagCollection>` object is immutable.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create TagMetas (see class TagMeta for additional information about creating TagMetas)
        meta_weather = sly.TagMeta('Weather', sly.TagValueType.ANY_STRING)

        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        meta_season = sly.TagMeta('Season', sly.TagValueType.ONEOF_STRING, possible_values=seasons)

        # Create Pointcloud Tags
        tag_weather = sly.PointcloudTag(meta_weather, value="Sunny")
        tag_season = sly.PointcloudTag(meta_season, value="Spring")
        tags_arr = [tag_weather, tag_season]

        # Create PointcloudTagCollection from Pointcloud Tags
        tags = sly.PointcloudTagCollection(tags_arr)

        # Add item to PointcloudTagCollection
        meta_potato = sly.TagMeta('potato', sly.TagValueType.NONE)
        tag_potato = sly.PointcloudTag(meta_potato)

        # Remember that PointcloudTagCollection is immutable, and we need to assign new instance of PointcloudTagCollection to a new variable
        tags = tags.add(tag_potato)

        # You can also add multiple items to collection
        meta_cabbage = sly.TagMeta('cabbage', sly.TagValueType.NONE)
        meta_carrot = sly.TagMeta('carrot', sly.TagValueType.NONE)
        meta_turnip = sly.TagMeta('turnip', sly.TagValueType.NONE)

        tag_cabbage = sly.PointcloudTag(meta_cabbage)
        tag_carrot = sly.PointcloudTag(meta_carrot)
        tag_turnip = sly.PointcloudTag(meta_turnip)

        additional_veggies = [tag_cabbage, tag_carrot, tag_turnip]

        tags = tags.add_items(additional_veggies)

        # Has key, checks if given key exist in collection
        tags.has_key(tag_cabbage.key())
        # Output: True

        # Intersection, finds intersection of given list of instances with collection items
        meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
        tag_dog = sly.PointcloudTag(meta_dog)

        meta_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
        tag_cat = sly.PointcloudTag(meta_cat)

        meta_turtle = sly.TagMeta('turtle', sly.TagValueType.NONE)
        tag_turtle = sly.PointcloudTag(meta_turtle)


        tags_animals = sly.PointcloudTagCollection([tag_dog, tag_cat, tag_turtle])

        tags_intersections = tags.intersection(tags_animals)
        print(tags_intersections.to_json())
        # Output: []

        # Let's add the potato Tag from another collection and compare them again
        tags_animals = tags_animals.add(tag_potato)

        tags_intersections = tags.intersection(tags_animals)
        print(tags_intersections.to_json())
        # Output: [
        #     {
        #         "name":"potato",
        #         "key": "058ad7993a534082b4d94cc52542a97d"
        #     }
        # ]

        # Difference, finds difference between collection and given list of Tags or TagCollection
        meta_car = sly.TagMeta('car', sly.TagValueType.NONE)
        tag_car = sly.Tag(meta_car)

        meta_bicycle = sly.TagMeta('bicycle', sly.TagValueType.NONE)
        tag_bicycle = sly.PointcloudTag(meta_bicycle)

        tags_vehicles = sly.PointcloudTagCollection([tag_car, tag_bicycle])

        meta_pedestrian = sly.TagMeta('pedestrian', sly.TagValueType.NONE)
        tag_pedestrian = sly.PointcloudTag(meta_pedestrian)

        meta_road = sly.TagMeta('road', sly.TagValueType.NONE)
        tag_road = sly.PointcloudTag(meta_road)

        difference = tags_vehicles.difference([tag_pedestrian, tag_road])
        print(difference.to_json())
        # Output: [
        #     {
        #         "name": "car",
        #         "key": "058ad7993a534082b4d94cc52542a97d"
        #     },
        #     {
        #         "name": "bicycle",
        #         "key": ""0c0033c5b4834d4cbabece4317295f07"
        #     }
        # ]

        # Merge, merges collection and given list of TagMetas
        tags_vehicles = sly.TagMetaCollection([tag_car, tag_bicycle])
        tags_merge = sly.TagMetaCollection([tag_pedestrian, tag_road])

        merged_collections = tags_vehicles.merge(tags_merge)
        print(merged_collections.to_json())
        # Output: [
        #     {
        #         "name":"pedestrian"
        #     },
        #     {
        #         "name":"road"
        #     },
        #     {
        #         "name":"car"
        #     },
        #     {
        #         "name":"bicycle"
        #     }
        # ]
    """

    item_type = PointcloudTag

    def __iter__(self) -> Iterator[PointcloudTag]:
        return next(self)

    # @classmethod
    # def from_api_response(
    #     cls,
    #     data: List[Dict],
    #     tag_meta_collection: TagMetaCollection,
    #     id_to_tagmeta: Optional[Dict[int, TagMeta]] = None
    # ) -> PointcloudTagCollection:
    #     return super().from_api_response(data, tag_meta_collection, id_to_tagmeta)

    # def get_by_name(self, tag_name: str) -> List[PointcloudTag]:
    #     """
    #     Get list of Pointcloud Tags with provided name.

    #     :param tag_name: Pointcloud Tag name.
    #     :type tag_name: :class:`str`
    #     :return: List of Pointcloud Tags.
    #     :rtype: :class:`List[PointcloudTag]<supervisely.pointcloud_annotation.pointcloud_tag.PointcloudTag>`
    #     """
    #     res = []
    #     for tag in self:
    #         if tag.name == tag_name:
    #             res.append(tag)
    #     return res

    # def get_single_by_name(self, tag_name: str) -> PointcloudTag:
    #     """
    #     Get Pointcloud Tag with provided name.
    #     Method will raise error If collection contains more than 1 tag with provided name.

    #     :param tag_name: Pointcloud Tag name.
    #     :type tag_name: :class:`str`
    #     :return: PointcloudTag object or :class:`None<None>` If no elements with provided name in collection.
    #     :rtype: :class:`PointcloudTag<supervisely.pointcloud_annotation.pointcloud_tag.PointcloudTag>` or :class:`NoneType<NoneType>`
    #     :raises: :class:`ValueError`, If collection contains more than 1 tag with provided name.
    #     """
    #     res = []
    #     for tag in self:
    #         if tag.name == tag_name:
    #             res.append(tag)
    #     if len(res) == 0:
    #         return None
    #     if len(res) > 1:
    #         raise ValueError(
    #             f"There are more than one tag {tag_name} in VideoTagCollection. Use method get_by_name instead"
    #         )
    #     return res[0]

    def to_json(self, key_id_map: Optional[KeyIdMap] = None) -> List[Dict]:
        """
        Convert the :class:`PointcloudTagCollection<PointcloudTagCollection>` to a list of json dicts.
        Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: List of dicts in json format
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta_weather = sly.TagMeta('Weather', sly.TagValueType.ANY_STRING)

            seasons = ["Winter", "Spring", "Summer", "Autumn"]
            meta_season = sly.TagMeta('Season', sly.TagValueType.ONEOF_STRING, possible_values=seasons)

            tag_weather = sly.PointcloudTag(meta_weather, value="Sunny")
            tag_season = sly.PointcloudTag(meta_season, value="Spring")

            tags_arr = [tag_weather, tag_season]
            tags = sly.PointcloudTagCollection(tags_arr)

            tags_json = tags.to_json()
            # Output: [
            #     {
            #         "name":"Weather",
            #         "value":"Sunny",
            #         "key": "058ad7993a534082b4d94cc52542a97d"
            #     },
            #     {
            #         "name":"Season",
            #         "value":"Spring",
            #         "key": "0c0033c5b4834d4cbabece4317295f07"
            #     }
            # ]
        """

        return [tag.to_json(key_id_map) for tag in self]

    @classmethod
    def from_json(
        cls,
        data: List[Dict],
        tag_meta_collection: TagMetaCollection,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> PointcloudTagCollection:
        """
        Convert a list with dicts in json format to :class:`PointcloudTagCollection<PointcloudTagCollection>`.
        Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: :class:`List[Dict]`
        :param tag_meta_collection: TagMetaCollection object.
        :type tag_meta_collection: :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: TagCollection object.
        :rtype: :class:`PointcloudTagCollection<PointcloudTagCollection>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Initialize TagMetaCollection

            meta_weather = sly.TagMeta('Weather', sly.TagValueType.ANY_STRING)

            seasons = ["Winter", "Spring", "Summer", "Autumn"]
            meta_season = sly.TagMeta('Season', sly.TagValueType.ONEOF_STRING, possible_values=seasons)

            metas_arr = [meta_weather, meta_season]
            tag_metas = sly.TagMetaCollection(metas_arr)

            data = [
                {
                    "name":"Weather",
                    "value":"Sunny"
                },
                {
                    "name":"Season",
                    "value":"Spring"
                }
            ]

            tags = sly.PointcloudTagCollection.from_json(data, tag_metas)
        """

        tags = [
            cls.item_type.from_json(tag_json, tag_meta_collection, key_id_map) for tag_json in data
        ]
        return cls(tags)
