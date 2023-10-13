from __future__ import annotations
from typing import List, Dict, Optional, Any, Iterator
from supervisely.pointcloud_annotation.pointcloud_tag_collection import PointcloudTagCollection
from supervisely.pointcloud_annotation.pointcloud_episode_tag import PointcloudEpisodeTag
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.annotation.tag_meta import TagMeta
from supervisely.video_annotation.key_id_map import KeyIdMap

class PointcloudEpisodeTagCollection(PointcloudTagCollection):
    """
    Collection with :class:`PointcloudEpisodeTag<supervisely.pointcloud_annotation.pointcloud_episode_tag.PointcloudEpisodeTag>` instances.
    :class:`PointcloudEpisodeTagCollection<PointcloudEpisodeTagCollection>` object is immutable.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create TagMetas (see class TagMeta for additional information about creating TagMetas)
        colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
        meta_car_color = sly.TagMeta('car color', sly.TagValueType.ONEOF_STRING, possible_values=colors)


        # Create tags
        tag_car_color_white = sly.PointcloudEpisodeTag(meta_car_color, value="white", frame_range=(15, 20))
        tag_car_color_red = sly.PointcloudEpisodeTag(meta_car_color, value="red", frame_range=(11, 22))
        tags_arr = [tag_car_color_white, tag_car_color_red]

        # Create PointcloudEpisodeTagCollection from point cloud episodes Tags
        tags = sly.PointcloudEpisodeTagCollection(tags_arr)

        # Add item to PointcloudEpisodeTagCollection
        meta_bus = sly.TagMeta('bus', sly.TagValueType.NONE)
        tag_bus = sly.PointcloudEpisodeTag(meta_bus, frame_range=(11, 14))

        # Remember that PointcloudEpisodeTagCollection is immutable, and we need to assign new instance of PointcloudEpisodeTagCollection to a new variable
        tags = tags.add(tag_bus)

        # You can also add multiple items to collection
        meta_truck = sly.TagMeta('truck', sly.TagValueType.NONE)
        meta_moto = sly.TagMeta('moto', sly.TagValueType.NONE)

        tag_truck = sly.PointcloudEpisodeTag(meta_truck, frame_range=(6, 10))
        tag_moto = sly.PointcloudEpisodeTag(meta_moto, frame_range=(11, 15))

        additional_tags = [tag_truck, tag_moto]

        tags = tags.add_items(additional_tags)

        # Has key, checks if given key exist in collection
        tags.has_key(meta_moto.key())
        # Output: True

        # Intersection, finds intersection of given list of instances with collection items
        meta_bus = sly.TagMeta('bus', sly.TagValueType.NONE)
        tag_bus = sly.PointcloudEpisodeTag(meta_bus, frame_range=(0, 5))

        meta_truck = sly.TagMeta('truck', sly.TagValueType.NONE, frame_range=(6, 11))
        tag_truck = sly.PointcloudEpisodeTag(meta_truck)

        tags_vehicles = sly.PointcloudEpisodeTagCollection([tag_bus, tag_truck])

        tags_intersections = tags.intersection(tags_vehicles)
        print(tags_intersections.to_json())
        # Output: []

        # Let's add the moto Tag from another collection and compare them again
        tags_vehicles = tags_vehicles.add(tag_moto)

        tags_intersections = tags.intersection(tags_vehicles)
        print(tags_intersections.to_json())
        # Output: [
        #     {
        #         "name": "moto",
        #         "frameRange": [11, 15],
        #         "key": "3f37e10a658e440db3ab9f5c0be7fb67"
        #     }
        # ]


        # Difference, finds difference between collection and given list of Tags or PointcloudEpisodeTagCollection
        meta_car = sly.TagMeta('car', sly.TagValueType.NONE)
        tag_car = sly.PointcloudEpisodeTag(meta_car, frame_range=(6, 11))

        meta_bicycle = sly.TagMeta('bicycle', sly.TagValueType.NONE)
        tag_bicycle = sly.PointcloudEpisodeTag(meta_bicycle, frame_range=(6, 11))

        tags_vehicles = sly.PointcloudEpisodeTagCollection([tag_car, tag_bicycle])

        meta_pedestrian = sly.TagMeta('pedestrian', sly.TagValueType.NONE)
        tag_pedestrian = sly.PointcloudEpisodeTag(meta_pedestrian, frame_range=(14, 20))

        meta_road = sly.TagMeta('road', sly.TagValueType.NONE)
        tag_road = sly.PointcloudEpisodeTag(meta_road, frame_range=(13, 22))

        difference = tags_vehicles.difference([tag_pedestrian, tag_road])
        print(difference.to_json())
        # Output: [
        #     {
        #         "name": "car",
        #         "frameRange": [6, 11],
        #         "key": "eba949cb996e44ac8aebec72a0875b13"
        #     },
        #     {
        #         "name": "bicycle",
        #         "frameRange": [6, 11],
        #         "key": "f9f03baec21e428ba04525d8df003e22"
        #     }
        # ]


        # Merge, merges collection and given list of collections
        tags_vehicles = sly.PointcloudEpisodeTagCollection([tag_car, tag_bicycle])
        tags_merge = sly.PointcloudEpisodeTagCollection([tag_pedestrian, tag_road])

        merged_collections = tags_vehicles.merge(tags_merge)
        print(merged_collections.to_json())
        # Output: [
        #     {
        #         "frameRange": [6, 11],
        #         "key": "a6527453a3d34e68865bc0ad4b66f673",
        #         "name": "car"
        #     },
        #     {
        #         "frameRange": [6, 11],
        #         "key": "e9a9a306fd9c4ea180d1edf2a6fb1cf0",
        #         "name": "bicycle"
        #     },
        #     {
        #         "frameRange": [14, 20],
        #         "key": "8480532ff4db4600839ab403b1d0ab85",
        #         "name": "pedestrian"
        #     },
        #     {
        #         "frameRange": [13, 22],
        #         "key": "828f58c0c5174dbba40f7374dcc8016a",
        #         "name": "road"
        #     }
        # ]
    """

    item_type = PointcloudEpisodeTag

    def __iter__(self) -> Iterator[PointcloudEpisodeTag]:
        return next(self)

    @classmethod
    def from_api_response(
        cls, 
        data: List[Dict], 
        tag_meta_collection: TagMetaCollection, 
        id_to_tagmeta: Optional[Dict[int, TagMeta]] = None
    ) -> PointcloudEpisodeTagCollection:
        """
        Create a PointcloudEpisodeTagCollection object from API response data.

        :param data: API response data.
        :type data: List[Dict]
        :param tag_meta_collection: TagMetaCollection object
        :type tag_meta_collection: TagMetaCollection
        :param id_to_tagmeta: Mapping of tag IDs to tag metadata.
        :type id_to_tagmeta: Optional[Dict[int, TagMeta]]
        :return: PointcloudEpisodeTagCollection object.
        :rtype: PointcloudEpisodeTagCollection
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

            project_id = 18428
            pcd_id = 19481209
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)
            pcd_info = api.pointcloud_episode.get_info_by_id(pcd_id)

            tag_collection = sly.PointcloudEpisodeTagCollection.from_api_response(
                pcd_info.tags, project_meta.tag_metas
            )
            print(tag_collection)

            # Output:
            # Tags:
            # +-------+--------------+-------+
            # |  Name |  Value type  | Value |
            # +-------+--------------+-------+
            # | color | oneof_string |  red  |
            # |  car  |     none     |  None |
            # +-------+--------------+-------+
        """

        return super().from_api_response(data, tag_meta_collection, id_to_tagmeta=id_to_tagmeta)

    # def get_by_name(self, tag_name: str) -> List[PointcloudEpisodeTag]:
    #     return super().get_by_name(tag_name)

    # def get_single_by_name(self, tag_name: str) -> PointcloudEpisodeTag:
    #     return super().get_single_by_name(tag_name)

    @classmethod
    def from_json(
        cls, 
        data: List[Dict], 
        tag_meta_collection: TagMetaCollection, 
        key_id_map: Optional[KeyIdMap] = None,
    ) -> PointcloudEpisodeTagCollection:
        """
        Convert a list with dicts in json format to :class:`PointcloudEpisodeTagCollection<PointcloudEpisodeTagCollection>`.
        Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: :class:`List[Dict]`
        :param tag_meta_collection: TagMetaCollection object.
        :type tag_meta_collection: :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: TagCollection object.
        :rtype: :class:`PointcloudEpisodeTagCollection<PointcloudEpisodeTagCollection>`
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

            tags = sly.PointcloudEpisodeTagCollection.from_json(data, tag_metas)
        """

        return super().from_json(data, tag_meta_collection, key_id_map=key_id_map)