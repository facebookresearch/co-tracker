# coding: utf-8
"""Collection with :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>` instances"""

# docs
from __future__ import annotations
from typing import List, Dict, Optional, Iterator
from supervisely.annotation.renamer import Renamer


from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.io.json import JsonSerializable
from supervisely.annotation.tag_meta import TagMeta


class TagMetaCollection(KeyIndexedCollection, JsonSerializable):
    """
    Collection with :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>` instances. :class:`TagMetaCollection<TagMetaCollection>` object is immutable.

    :raises: :class:`DuplicateKeyError`, if instance with given name already exists
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create TagMetas
        meta_weather = sly.TagMeta('Weather', sly.TagValueType.ANY_STRING)

        season_values = ["Winter", "Spring", "Summer", "Autumn"]
        meta_season = sly.TagMeta('Season', sly.TagValueType.ONEOF_STRING, possible_values=season_values)

        # Create TagMetaCollection from TagMetas
        tag_metas = sly.TagMetaCollection([meta_weather, meta_season])

        # Add items to TagMetaCollection
        meta_potato = sly.TagMeta('potato', sly.TagValueType.NONE)

        # Remember that TagMetaCollection is immutable, and we need to assign new instance of TagMetaCollection to a new variable
        tag_metas = tag_metas.add(meta_potato)

        # You can also add multiple items to collection
        meta_cabbage = sly.TagMeta('cabbage', sly.TagValueType.NONE)
        meta_carrot = sly.TagMeta('carrot', sly.TagValueType.NONE)
        meta_turnip = sly.TagMeta('turnip', sly.TagValueType.NONE)

        tag_metas = tag_metas.add_items([meta_cabbage, meta_carrot, meta_turnip])

        # Has key, checks if given key exist in collection
        tag_metas.has_key("cabbage")
        # Output: True

        # Intersection, finds intersection of given list of instances with collection items
        meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
        meta_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
        meta_turtle = sly.TagMeta('turtle', sly.TagValueType.NONE)

        tag_metas_animals = sly.TagMetaCollection([meta_dog, meta_cat, meta_turtle])

        metas_intersections = tag_metas.intersection(tag_metas_animals)
        print(metas_intersections.to_json())
        # Output: []

        # Let's add the potato TagMeta from another collection and compare them again
        tag_metas_animals = tag_metas_animals.add(meta_potato)

        metas_intersections = tag_metas.intersection(tag_metas_animals)
        print(metas_intersections.to_json())
        # Output: [
        #     {
        #         "name":"potato",
        #         "value_type":"none",
        #         "color":"#8A710F",
        #         "hotkey":"",
        #         "applicable_type":"all",
        #         "classes":[]
        #     }
        # ]

        # Difference, finds difference between collection and given list of TagMetas or TagMetaCollection
        meta_car = sly.TagMeta('car', sly.TagValueType.NONE)
        meta_bicycle = sly.TagMeta('bicycle', sly.TagValueType.NONE)

        tag_metas_vehicles = sly.TagMetaCollection([meta_car, meta_bicycle])

        meta_pedestrian = sly.TagMeta('pedestrian', sly.TagValueType.NONE)
        meta_road = sly.TagMeta('road', sly.TagValueType.NONE)

        difference = tag_metas_vehicles.difference([meta_pedestrian, meta_road])
        print(difference.to_json())
        # Output: [
        #     {
        #         "name":"car",
        #         "value_type":"none",
        #         "color":"#0F138A",
        #         "hotkey":"",
        #         "applicable_type":"all",
        #         "classes":[]
        #     },
        #     {
        #         "name":"bicycle",
        #         "value_type":"none",
        #         "color":"#0F8A25",
        #         "hotkey":"",
        #         "applicable_type":"all",
        #         "classes":[]
        #     }
        # ]

        # Merge, merges collection and given list of TagMetas
        tag_metas_vehicles = sly.TagMetaCollection([meta_car, meta_bicycle])
        tag_metas_merge = sly.TagMetaCollection([meta_pedestrian, meta_road])

        merged_collections = tag_metas_vehicles.merge(tag_metas_merge)
        print(merged_collections.to_json())
        # Output: [
        #     {
        #         "name":"pedestrian",
        #         "value_type":"none",
        #         "color":"#698A0F",
        #         "hotkey":"",
        #         "applicable_type":"all",
        #         "classes":[]
        #     },
        #     {
        #         "name":"road",
        #         "value_type":"none",
        #         "color":"#0F8A59",
        #         "hotkey":"",
        #         "applicable_type":"all",
        #         "classes":[]
        #     },
        #     {
        #         "name":"car",
        #         "value_type":"none",
        #         "color":"#0F138A",
        #         "hotkey":"",
        #         "applicable_type":"all",
        #         "classes":[]
        #     },
        #     {
        #         "name":"bicycle",
        #         "value_type":"none",
        #         "color":"#0F8A25",
        #         "hotkey":"",
        #         "applicable_type":"all",
        #         "classes":[]
        #     }
        # ]

        # Merge will raise ValueError if item name from given list is in collection but items in both are different
        meta_bicycle_1 = sly.TagMeta('bicycle', sly.TagValueType.NONE)
        meta_bicycle_2 = sly.TagMeta('bicycle', sly.TagValueType.ANY_STRING)

        tag_metas_1 = sly.TagMetaCollection([meta_bicycle_1])
        tag_metas_2 = sly.TagMetaCollection([meta_bicycle_2])

        test_merge = tag_metas_1.merge(tag_metas_2)
        # Output: ValueError: Error during merge for key 'bicycle': values are different

        # Let's try to create now a collection where TagMetas have identical names
        meta_cow = sly.TagMeta('cow', sly.TagValueType.NONE)
        meta_chicken = sly.TagMeta('cow', sly.TagValueType.NONE)

        tag_metas = sly.TagMetaCollection([meta_cow, meta_chicken])
        # Output: DuplicateKeyError: "Key 'cow' already exists"
    """

    item_type = TagMeta

    def to_json(self) -> List[Dict]:
        """
        Convert the TagMetaCollection to a list of json dicts. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: List of dicts in json format
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            tag_metas = tag_metas.to_json()
            # Output:
            # [
            #   {
            #    "name": "Weather",
            #    "value_type": "any_string",
            #    "color": "#8A620F",
            #    "hotkey": "",
            #    "applicable_type": "all",
            #    "classes": []
            #   },
            #   {
            #    "name": "Season",
            #    "value_type": "oneof_string",
            #    "color": "#700F8A",
            #    "values": ["Winter", 'Spring", "Summer", "Autumn"],
            #    "hotkey": "",
            #    "applicable_type": "all",
            #    "classes": []
            #   }
            # ]
        """
        return [tag_meta.to_json() for tag_meta in self]

    @classmethod
    def from_json(cls, data: List[Dict]) -> TagMetaCollection:
        """
        Convert a list with dicts in json format to TagMetaCollection. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: List[dict]
        :return: TagMetaCollection object
        :rtype: :class:`TagMetaCollection<TagMetaCollection>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            data = [
                {
                    "name":"Weather",
                    "value_type":"any_string",
                    "color":"#8A620F",
                    "hotkey":"",
                    "applicable_type":"all",
                    "classes":[]
                },
                {
                    "name":"Season",
                    "value_type": "oneof_string",
                                  "color": "#700F8A",
                                  "values": ["Winter", "Spring", "Summer", "Autumn"],
                    "hotkey":"",
                    "applicable_type":"all",
                    "classes":[]
            }
            ]

            tag_metas = sly.TagMetaCollection.from_json(data)
        """
        tags = [TagMeta.from_json(tag_meta_json) for tag_meta_json in data]
        return cls(tags)

    def get_id_mapping(self, raise_if_no_id: Optional[bool] = False) -> Dict[int, TagMeta]:
        """
        Create dict matching TagMetas id to TagMeta.

        :param raise_if_no_id: Raise ValueError if where is TagMetas without id.
        :type raise_if_no_id: bool, optional
        :return: Json format as a dict
        :rtype: :class:`dict`
        :raises: :class:`KeyError`, if where is duplication of TagMetas id
        """
        res = {}
        without_id = []
        for tag_meta in self:
            if tag_meta.sly_id is not None:
                if tag_meta.sly_id in res:
                    raise KeyError(
                        f"TagMeta with id={tag_meta.sly_id} already exists (duplication). "
                        f"Please contact tech support"
                    )
                else:
                    res[tag_meta.sly_id] = tag_meta
            else:
                without_id.append(tag_meta)
        if len(without_id) > 0 and raise_if_no_id is True:
            raise ValueError("There are TagMetas without id")
        return res

    def __iter__(self) -> Iterator[TagMeta]:
        return next(self)

    def refresh_ids_from(self, tags: TagMetaCollection) -> None:
        for new_tag in tags:
            my_tag = self.get(new_tag.name)
            if my_tag is None:
                continue
            my_tag._set_id(new_tag.sly_id)

    def get_by_id(self, tag_meta_id: int) -> TagMeta:
        for tag_meta in self:
            if tag_meta.sly_id == tag_meta_id:
                return tag_meta
        return None


def make_renamed_tag_metas(
    src_tag_metas: TagMetaCollection, renamer: Renamer, skip_missing: bool = False
) -> TagMetaCollection:
    """make_renamed_tag_metas"""
    result_tags = []
    for src_tag in src_tag_metas:
        renamed_name = renamer.rename(src_tag.name)
        if renamed_name is not None:
            result_tags.append(src_tag.clone(name=renamed_name))
        elif not skip_missing:
            raise KeyError(
                "Tag meta named {} could not be mapped to a destination name.".format(src_tag.name)
            )
    return TagMetaCollection(items=result_tags)
