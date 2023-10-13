# coding: utf-8


# docs
from __future__ import annotations
from typing import List, Dict, Optional, Iterator, Any
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.annotation.tag_meta import TagMeta
from supervisely.annotation.tag_collection import TagCollection
from supervisely.video_annotation.video_tag import VideoTag


class VideoTagCollection(TagCollection):
    """
    Collection with :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` instances. :class:`VideoTagCollection<VideoTagCollection>` object is immutable.

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.video_annotation.video_tag import VideoTag
        from supervisely.video_annotation.video_tag_collection import VideoTagCollection

        # Create two VideoTags for collection
        meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
        car_tag = VideoTag(meta_car, value='acura')
        meta_bus = sly.TagMeta('bus_tag', sly.TagValueType.ANY_STRING)
        bus_tag = VideoTag(meta_bus, value='volvo')

        # Create VideoTagCollection
        tags = VideoTagCollection([car_tag, bus_tag])
        tags_json = tags.to_json()
        print(tags_json)
        # Output: [
        #     {
        #         "name": "car_tag",
        #         "value": "acura",
        #         "key": "378408fcb6854305a38fed7c996f4901"
        #     },
        #     {
        #         "name": "bus_tag",
        #         "value": "volvo",
        #         "key": "0c63174878204faea67c4025adec1e2a"
        #     }
        # ]

        # Add item to VideoTagCollection
        meta_truck = sly.TagMeta('truck_tag', sly.TagValueType.NONE)
        truck_tag = VideoTag(meta_truck)
        # Remember that VideoTagCollection is immutable, and we need to assign new instance of VideoTagCollection to a new variable
        new_tags = tags.add(truck_tag)
        new_tags_json = new_tags.to_json()
        print(new_tags_json)
        # Output: [
        #     {
        #         "name": "car_tag",
        #         "value": "acura",
        #         "key": "3d33f1685dab44da9b55d67bab3937e9"
        #     },
        #     {
        #         "name": "bus_tag",
        #         "value": "volvo",
        #         "key": "7c52c168a22b47a5b39fbcdef94b0140"
        #     },
        #     {
        #         "name": "truck_tag",
        #         "key": "7188242d230b4d2783c588cc2eca5ff8"
        #     }
        # ]

        # You can also add multiple items to collection
        meta_truck = sly.TagMeta('truck_tag', sly.TagValueType.NONE)
        truck_tag = VideoTag(meta_truck)
        meta_train = sly.TagMeta('train_tag', sly.TagValueType.ANY_NUMBER)
        train_tag = VideoTag(meta_train, value=777)
        new_tags = tags.add_items([truck_tag, train_tag])
        new_tags_json = new_tags.to_json()
        print(new_tags_json)
        # Output: [
        #     {
        #         "name": "car_tag",
        #         "value": "acura",
        #         "key": "03b052c464d84d5db6451b86f3bdef79"
        #     },
        #     {
        #         "name": "bus_tag",
        #         "value": "volvo",
        #         "key": "0f7dd8fb6f8c41da9e68e64d3186df15"
        #     },
        #     {
        #         "name": "truck_tag",
        #         "key": "fc01af7c70154771b7253b6a94484179"
        #     },
        #     {
        #         "name": "train_tag",
        #         "value": 777,
        #         "key": "6ce5118181074a52b74dee9335fa292d"
        #     }
        # ]

        # Intersection, finds intersection of given list of VideoTag instances with collection items
        intersect_tags = tags.intersection([bus_tag])
        intersect_tags_json = intersect_tags.to_json()
        print(intersect_tags_json)
        # Output: [
        #     {
        #         "name": "bus_tag",
        #         "value": "volvo",
        #         "key": "13d77d8c848e4a3ebeb710bf5f3f38a6"
        #     }
        # ]

        # Difference, finds difference between collection and given list of VideoTag
        diff_tags = tags.difference([bus_tag])
        diff_tags_json = diff_tags.to_json()
        print(diff_tags_json)
        # Output: [
        #     {
        #         "name": "car_tag",
        #         "value": "acura",
        #         "key": "341b9fc077e142c0956d5cf985d705c1"
        #     }
        # ]

        # Merge, merges collection and given list of VideoTagCollection
        meta_truck = sly.TagMeta('truck_tag', sly.TagValueType.NONE)
        truck_tag = VideoTag(meta_truck)
        meta_train = sly.TagMeta('train_tag', sly.TagValueType.ANY_NUMBER)
        train_tag = VideoTag(meta_train, value=777)
        over_tags = VideoTagCollection([truck_tag, train_tag])
        # Merge
        merge_tags = tags.merge(over_tags)
        merge_tags_json = merge_tags.to_json()
        print(merge_tags_json)
        # Output: [
        #     {
        #         "name": "car_tag",
        #         "value": "acura",
        #         "key": "3bfde60091a24dd493485f2b80364736"
        #     },
        #     {
        #         "name": "bus_tag",
        #         "value": "volvo",
        #         "key": "dac5b07954064c78acadfa71161f6998"
        #     },
        #     {
        #         "name": "truck_tag",
        #         "key": "a46aac4499574afc81d41d2bb061a3c6"
        #     },
        #     {
        #         "name": "train_tag",
        #         "value": 777,
        #         "key": "bf5b1667383449478694fb6349d7b16c"
        #     }
        # ]
    """

    item_type = VideoTag

    def to_json(self, key_id_map: Optional[KeyIdMap] = None) -> List[Dict]:
        """
        Convert the VideoTagCollection to a list of json dicts. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: List of dicts in json format
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.video_tag import VideoTag
            from supervisely.video_annotation.video_tag_collection import VideoTagCollection

            meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
            car_tag = VideoTag(meta_car, value='acura')
            meta_bus = sly.TagMeta('bus_tag', sly.TagValueType.ANY_STRING)
            bus_tag = VideoTag(meta_bus, value='volvo')
            tags = VideoTagCollection([car_tag, bus_tag])
            tags_json = tags.to_json()
            print(tags_json)
            # Output: [
            #     {
            #         "name": "car_tag",
            #         "value": "acura",
            #         "key": "378408fcb6854305a38fed7c996f4901"
            #     },
            #     {
            #         "name": "bus_tag",
            #         "value": "volvo",
            #         "key": "0c63174878204faea67c4025adec1e2a"
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
    ) -> VideoTagCollection:
        """
        Convert a list of json dicts to VideoTagCollection. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: List[dict]
        :param project_meta: Input TagMetaCollection object.
        :type project_meta: TagMetaCollection
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: VideoTagCollection object
        :rtype: :class:`VideoTagCollection`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video_annotation.video_tag_collection import VideoTagCollection

            tags_json = [
                {
                    "name": "car_tag",
                    "value": "acura",
                },
                {
                    "name": "bus_tag",
                    "value": "volvo",
                }
            ]
            meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
            meta_bus = sly.TagMeta('bus_tag', sly.TagValueType.ANY_STRING)
            meta_collection = sly.TagMetaCollection([meta_car, meta_bus])

            tags = VideoTagCollection.from_json(tags_json, meta_collection)
        """
        tags = [
            cls.item_type.from_json(tag_json, tag_meta_collection, key_id_map) for tag_json in data
        ]
        return cls(tags)

    def __iter__(self) -> Iterator[VideoTag]:
        return next(self)

    @classmethod
    def from_api_response(
        cls,
        data: List[Dict],
        tag_meta_collection: TagMetaCollection,
        id_to_tagmeta: Optional[Dict[int, TagMeta]] = None,
    ) -> VideoTagCollection:
        """
        Create a VideoTagCollection object from API response data.

        :param data: API response data.
        :type data: List[Dict]
        :param tag_meta_collection: _description_
        :type tag_meta_collection: TagMetaCollection
        :param id_to_tagmeta: Mapping of tag IDs to tag metadata.
        :type id_to_tagmeta: Optional[Dict[int, TagMeta]]
        :return: VideoTagCollection object.
        :rtype: VideoTagCollection
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

            videos = api.video.get_list(dataset_id)
            for info in videos:
                tag_collection = sly.VideoTagCollection.from_api_response(
                    info.tags, project_meta.tag_metas
                )
        """
        return super().from_api_response(data, tag_meta_collection, id_to_tagmeta)

    def get_by_name(self, tag_name: str, default: Optional[Any] = None) -> List[VideoTag]:
        """
        Get a list of VideoTag objects by name from the VideoTagCollection.

        :param tag_name: Name of the tags to get.
        :type tag_name: str
        :return:  List of VideoTag objects with the specified name.
        :rtype: List[VideoTag]
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

            videos = api.video.get_list(dataset_id)
            for info in videos:
                tag_collection = sly.VideoTagCollection.from_api_response(
                    info.tags, project_meta.tag_metas
                )
                single_tag = tag_collection.get_by_name("tag_name")
        """
        # super().get_by_name(tag_name, default)
        res = []
        for tag in self:
            if tag.name == tag_name:
                res.append(tag)
        return res

    def get_single_by_name(self, tag_name: str, default: Optional[Any] = None) -> VideoTag:
        """
        Get a single Tag object by name from the VideoTagCollection.

        :param tag_name: Name of the tag to get.
        :type tag_name: str
        :return: VideoTag object with the specified name.
        :rtype: VideoTag
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

            videos = api.video.get_list(dataset_id)
            for info in videos:
                tag_collection = sly.VideoTagCollection.from_api_response(
                    info.tags, project_meta.tag_metas
                )
                single_tag = tag_collection.get_single_by_name("tag_name")
        """
        # super().get_by_name(tag_name, default)
        res = []
        for tag in self:
            if tag.name == tag_name:
                res.append(tag)
        if len(res) == 0:
            return None
        if len(res) > 1:
            raise ValueError(
                f"There are more than one tag {tag_name} in VideoTagCollection. Use method get_by_name instead"
            )
        return res[0]
