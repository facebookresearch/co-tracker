# coding: utf-8
from supervisely.annotation.renamer import Renamer
from supervisely.annotation.tag_collection import TagCollection
from supervisely.annotation.tag_meta import TagMeta
from supervisely.annotation.tag_meta_collection import TagMetaCollection


class TagMetaMapper:
    """
    """

    def map(self, src: TagMeta) -> TagMeta:
        """
        """
        raise NotImplementedError()


class RenamingTagMetaMapper(TagMetaMapper):
    """
    This is a class for renaming TagMeta in given TagMetaCollection
    """

    def __init__(self, dest_tag_meta_dict: TagMetaCollection, renamer: Renamer):
        self._dest_tag_meta_dict = dest_tag_meta_dict
        self._renamer = renamer

    def map(self, src: TagMeta) -> TagMeta:
        """
        The function map rename TagMeta in given collection
        :return: TagMeta
        """
        dest_name = self._renamer.rename(src.name)
        return self._dest_tag_meta_dict.get(dest_name, None) if (dest_name is not None) else None


def make_renamed_tags(tags: TagCollection, tag_meta_mapper: TagMetaMapper, skip_missing=True) -> TagCollection:
    """
    The function make_renamed_tags rename tags names in given collection and return new collection
    :return: TagCollection
    """
    renamed_tags = []
    for tag in tags:
        dest_tag_meta = tag_meta_mapper.map(tag.meta)
        if dest_tag_meta is not None:
            renamed_tags.append(tag.clone(meta=dest_tag_meta))
        elif not skip_missing:
            raise KeyError('Tag named {} could not be mapped to a destination name.'.format(tag.meta.name))
    return TagCollection(items=renamed_tags)
