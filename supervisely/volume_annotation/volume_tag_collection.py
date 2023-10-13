# coding: utf-8

from supervisely.video_annotation.video_tag_collection import VideoTagCollection
from supervisely.volume_annotation.volume_tag import VolumeTag


class VolumeTagCollection(VideoTagCollection):
    """
    Collection with :class:`VolumeTag<supervisely.volume_annotation.volume_tag.VolumeTag>` instances. :class:`VolumeTagCollection<VolumeTagCollection>` object is immutable.

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.volume_annotation.volume_tag import VolumeTag
        from supervisely.volume_annotation.volume_tag_collection import VolumeTagCollection

        # Create two VolumeTags for collection
        meta_brain = sly.TagMeta('brain_tag', sly.TagValueType.ANY_STRING)
        brain_tag = VolumeTag(meta_brain, value='brain')
        meta_heart = sly.TagMeta('heart_tag', sly.TagValueType.ANY_STRING)
        heart_tag = VolumeTag(meta_heart, value='heart')

        # Create VolumeTagCollection
        tags = VolumeTagCollection([brain_tag, heart_tag])
        tags_json = tags.to_json()
        print(tags_json)
        # Output:
        # [
        #     {
        #         "key": "9fbbc3f888594a538243445fe25242ec",
        #         "name": "brain_tag",
        #         "value": "brain"
        #     },
        #     {
        #         "key": "804c8124d46c4da89b54c6132acf06a0",
        #         "name": "heart_tag",
        #         "value": "heart"
        #     }
        # ]


        # Add item to VolumeTagCollection
        meta_lang = sly.TagMeta('lang_tag', sly.TagValueType.NONE)
        lang_tag = VolumeTag(meta_lang)
        # Remember that VolumeTagCollection is immutable, and we need to assign new instance of VolumeTagCollection to a new variable
        new_tags = tags.add(lang_tag)
        new_tags_json = new_tags.to_json()
        print(new_tags_json)
        # Output: [
        #     {
        #     {
        #         "key": "9fbbc3f888594a538243445fe25242ec",
        #         "name": "brain_tag",
        #         "value": "brain"
        #     },
        #     {
        #         "key": "804c8124d46c4da89b54c6132acf06a0",
        #         "name": "heart_tag",
        #         "value": "heart"
        #     },
        #     {
        #         "key": "7188242d2ddb4d2783c588cc2eca5ff8",
        #         "name": "lang_tag"
        #     }
        # ]

        # You can also add multiple items to collection
        meta_leg = sly.TagMeta('leg_tag', sly.TagValueType.NONE)
        leg_tag = VolumeTag(meta_leg)
        meta_arm = sly.TagMeta('arm_tag', sly.TagValueType.ANY_NUMBER)
        arm_tag = VolumeTag(meta_arm, value=777)
        new_tags = tags.add_items([leg_tag, arm_tag])
        new_tags_json = new_tags.to_json()
        print(new_tags_json)
        # Output: [
        #     {
        #     {
        #         "key": "9fbbc3f888594a538243445fe25242ec",
        #         "name": "brain_tag",
        #         "value": "brain"
        #     },
        #     {
        #         "key": "804c8124d46c4da89b54c6132acf06a0",
        #         "name": "heart_tag",
        #         "value": "heart"
        #     },
        #     {
        #         "key": "7188242d2ddb4d2783c588cc2eca5ff8",
        #         "name": "lang_tag"
        #     },
        #     {
        #         "key": "111c8124d46c4da89b5lgk132acf06a0",
        #         "name": "leg_tag",
        #     },
        #     {
        #         "key": "28dc8124d46c4da89b54c6132acf06a0",
        #         "name": "arm_tag",
        #         "value": "777"
        #     },
        # ]

        # Intersection, finds intersection of given list of VolumeTag instances with collection items
        intersect_tags = tags.intersection([leg_tag])
        intersect_tags_json = intersect_tags.to_json()
        print(intersect_tags_json)
        # Output: [
        #     {
        #         "key": "111c8124d46c4da89b5lgk132acf06a0",
        #         "name": "leg_tag",
        #     }
        # ]

        # Difference, finds difference between collection and given list of VolumeTag
        diff_tags = tags.difference([leg_tag, arm_tag, lang_tag])
        diff_tags_json = diff_tags.to_json()
        print(diff_tags_json)
        # Output:
        # [
        #     {
        #         "key": "9fbbc3f888594a538243445fe25242ec",
        #         "name": "brain_tag",
        #         "value": "brain"
        #     },
        #     {
        #         "key": "804c8124d46c4da89b54c6132acf06a0",
        #         "name": "heart_tag",
        #         "value": "heart"
        #     }
        # ]

        # Merge, merges collection and given list of VolumeTagCollection
        meta_leg = sly.TagMeta('leg_tag', sly.TagValueType.NONE)
        leg_tag = VolumeTag(meta_leg)
        meta_arm = sly.TagMeta('arm_tag', sly.TagValueType.ANY_NUMBER)
        arm_tag = VolumeTag(meta_arm, value=777)
        over_tags = VolumeTagCollection([leg_tag, arm_tag])
        # Merge
        merge_tags = tags.merge(over_tags)
        merge_tags_json = merge_tags.to_json()
        print(merge_tags_json)
        # Output: [
        # [
        #     {
        #         "key": "9fbbc3f888594a538243445fe25242ec",
        #         "name": "brain_tag",
        #         "value": "brain"
        #     },
        #     {
        #         "key": "804c8124d46c4da89b54c6132acf06a0",
        #         "name": "heart_tag",
        #         "value": "heart"
        #     },
        #     {
        #         "key": "111c8124d46c4da89b5lgk132acf06a0",
        #         "name": "leg_tag",
        #     },
        #     {
        #         "key": "28dc8124d46c4da89b54c6132acf06a0",
        #         "name": "arm_tag",
        #         "value": "777"
        #     }
        # ]
    """

    item_type = VolumeTag
