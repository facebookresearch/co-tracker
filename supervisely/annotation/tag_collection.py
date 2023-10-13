# coding: utf-8
"""collection with :class:`Tag<supervisely.annotation.tag.Tag>` instances"""

# docs
from __future__ import annotations
from typing import List, Optional, Dict, Iterator, Any
from supervisely.annotation.tag_meta_collection import TagMetaCollection

from supervisely.collection.key_indexed_collection import MultiKeyIndexedCollection
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta


class TagCollection(MultiKeyIndexedCollection):
    """
    Collection with :class:`Tag<supervisely.annotation.tag.Tag>` instances. :class:`TagCollection<TagCollection>` object is immutable.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create TagMetas (see class TagMeta for additional information about creating TagMetas)
        meta_weather = sly.TagMeta('Weather', sly.TagValueType.ANY_STRING)

        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        meta_season = sly.TagMeta('Season', sly.TagValueType.ONEOF_STRING, possible_values=seasons)

        # Create Tags
        tag_weather = sly.Tag(meta_weather, value="Sunny")
        tag_season = sly.Tag(meta_season, value="Spring")
        tags_arr = [tag_weather, tag_season]

        # Create TagCollection from Tags
        tags = sly.TagCollection(tags_arr)

        # Add item to TagCollection
        meta_potato = sly.TagMeta('potato', sly.TagValueType.NONE)
        tag_potato =sly.Tag(meta_potato)

        # Remember that TagCollection is immutable, and we need to assign new instance of TagCollection to a new variable
        tags = tags.add(tag_potato)

        # You can also add multiple items to collection
        meta_cabbage = sly.TagMeta('cabbage', sly.TagValueType.NONE)
        meta_carrot = sly.TagMeta('carrot', sly.TagValueType.NONE)
        meta_turnip = sly.TagMeta('turnip', sly.TagValueType.NONE)

        tag_cabbage = sly.Tag(meta_cabbage)
        tag_carrot = sly.Tag(meta_carrot)
        tag_turnip = sly.Tag(meta_turnip)

        additional_veggies = [tag_cabbage, tag_carrot, tag_turnip]

        tags = tags.add_items(additional_veggies)

        # Has key, checks if given key exist in collection
        tags.has_key("cabbage")
        # Output: True

        # Intersection, finds intersection of given list of instances with collection items
        meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
        tag_dog = sly.Tag(meta_dog)

        meta_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
        tag_cat = sly.Tag(meta_cat)

        meta_turtle = sly.TagMeta('turtle', sly.TagValueType.NONE)
        tag_turtle = sly.Tag(meta_turtle)


        tags_animals = sly.TagCollection([tag_dog, tag_cat, tag_turtle])

        tags_intersections = tags.intersection(tags_animals)
        print(tags_intersections.to_json())
        # Output: []

        # Let's add the potato Tag from another collection and compare them again
        tags_animals = tags_animals.add(tag_potato)

        tags_intersections = tags.intersection(tags_animals)
        print(tags_intersections.to_json())
        # Output: [
        #     {
        #         "name":"potato"
        #     }
        # ]

        # Difference, finds difference between collection and given list of Tags or TagCollection
        meta_car = sly.TagMeta('car', sly.TagValueType.NONE)
        tag_car = sly.Tag(meta_car)

        meta_bicycle = sly.TagMeta('bicycle', sly.TagValueType.NONE)
        tag_bicycle = sly.Tag(meta_bicycle)

        tags_vehicles = sly.TagCollection([tag_car, tag_bicycle])

        meta_pedestrian = sly.TagMeta('pedestrian', sly.TagValueType.NONE)
        tag_pedestrian = sly.Tag(meta_pedestrian)

        meta_road = sly.TagMeta('road', sly.TagValueType.NONE)
        tag_road = sly.Tag(meta_road)

        difference = tags_vehicles.difference([tag_pedestrian, tag_road])
        print(difference.to_json())
        # Output: [
        #     {
        #         "name":"car"
        #     },
        #     {
        #         "name":"bicycle"
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

    item_type = Tag

    def to_json(self) -> List[Dict]:
        """
        Convert the TagCollection to a list of json dicts. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: List of dicts in json format
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta_weather = sly.TagMeta('Weather', sly.TagValueType.ANY_STRING)

            seasons = ["Winter", "Spring", "Summer", "Autumn"]
            meta_season = sly.TagMeta('Season', sly.TagValueType.ONEOF_STRING, possible_values=seasons)

            tag_weather = sly.Tag(meta_weather, value="Sunny")
            tag_season = sly.Tag(meta_season, value="Spring")

            tags_arr = [tag_weather, tag_season]
            tags = sly.TagCollection(tags_arr)

            tags_json = tags.to_json()
            # Output: [
            #     {
            #         "name":"Weather",
            #         "value":"Sunny"
            #     },
            #     {
            #         "name":"Season",
            #         "value":"Spring"
            #     }
            # ]
        """
        return [tag.to_json() for tag in self]

    @classmethod
    def from_json(cls, data: List[Dict], tag_meta_collection: TagMetaCollection) -> TagCollection:
        """
        Convert a list with dicts in json format to TagCollection. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: List[dict]
        :param tag_meta_collection: Input TagMetaCollection object.
        :type tag_meta_collection: TagMetaCollection
        :return: TagCollection object
        :rtype: :class:`TagCollection<TagCollection>`
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

            tags = sly.TagCollection.from_json(data, tag_metas)
        """
        tags = [cls.item_type.from_json(tag_json, tag_meta_collection) for tag_json in data]
        return cls(tags)

    def __str__(self):
        return "Tags:\n" + super(TagCollection, self).__str__()

    @classmethod
    def from_api_response(
        cls,
        data: List[Dict],
        tag_meta_collection: TagMetaCollection,
        id_to_tagmeta: Optional[Dict[int, TagMeta]] = None,
    ) -> TagCollection:
        """
        Create a TagCollection object from API response data.

        :param data: API response data.
        :type data: List[Dict]
        :param tag_meta_collection: TagMetaCollection object
        :type tag_meta_collection: TagMetaCollection
        :param id_to_tagmeta: Mapping of tag IDs to tag metadata.
        :type id_to_tagmeta: Optional[Dict[int, TagMeta]]
        :return: TagCollection object.
        :rtype: TagCollection
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # You can connect to API directly
            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Or you can use API from environment
            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 17200
            image_id = 19369643
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            image_info = api.image.get_info_by_id(image_id)

            tag_collection = sly.TagCollection.from_api_response(
                image_info.tags, project_meta.tag_metas
            )
            print(tag_collection)

            # Output:
            # Tags:
            # +-------+--------------+-------+
            # |  Name |  Value type  | Value |
            # +-------+--------------+-------+
            # | Lemon | oneof_string |  big  |
            # |  Kiwi |     none     |  None |
            # +-------+--------------+-------+
        """

        if id_to_tagmeta is None:
            id_to_tagmeta = tag_meta_collection.get_id_mapping()
        tags = []
        for tag_json in data:
            tag_meta_id = tag_json["tagId"]
            tag_meta = id_to_tagmeta[tag_meta_id]
            tag_json["name"] = tag_meta.name
            tag = cls.item_type.from_json(tag_json, tag_meta_collection)
            tags.append(tag)
        return cls(tags)

    def __iter__(self) -> Iterator[Tag]:
        return next(self)
