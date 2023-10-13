# coding: utf-8

# docs
from typing import List, Tuple, Optional, Dict, Callable, Union

from tqdm import tqdm

from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely._utils import batched


class AdvancedApi(ModuleApiBase):
    """class AdvancedApi"""

    def add_tag_to_object(
        self, tag_meta_id: int, figure_id: int, value: Optional[str or int] = None
    ) -> Dict:
        """add_tag_to_object"""
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.FIGURE_ID: figure_id}
        if value is not None:
            data[ApiField.VALUE] = value
        resp = self._api.post("object-tags.add-to-object", data)
        return resp.json()

    def remove_tag_from_object(self, tag_meta_id: int, figure_id: int, tag_id: int) -> Dict:
        """remove_tag_from_object"""
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.FIGURE_ID: figure_id, ApiField.ID: tag_id}
        resp = self._api.post("object-tags.remove-from-figure", data)
        return resp.json()

    def get_object_tags(self, figure_id: int) -> Dict:
        """get_object_tags"""
        data = {ApiField.ID: figure_id}
        resp = self._api.post("figures.tags.list", data)
        return resp.json()

    def remove_tag_from_image(self, tag_meta_id: int, image_id: int, tag_id: int) -> Dict:
        """remove_tag_from_image"""
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.IMAGE_ID: image_id, ApiField.ID: tag_id}
        resp = self._api.post("image-tags.remove-from-image", data)
        return resp.json()

    def remove_tags_from_images(
        self,
        tag_meta_ids: List[int],
        image_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """remove_tags_from_images"""
        for batch_ids in batched(image_ids, batch_size=100):
            data = {ApiField.TAG_IDS: tag_meta_ids, ApiField.IDS: batch_ids}
            self._api.post("image-tags.bulk.remove-from-images", data)
            if progress_cb is not None:
                progress_cb(len(batch_ids))
