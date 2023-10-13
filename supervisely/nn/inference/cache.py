import os
import shutil
import numpy as np

from cacheout import Cache as CacheOut
from cachetools import LRUCache, Cache, TTLCache
from time import sleep
from enum import Enum
from fastapi import Request, FastAPI
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
from threading import Lock, Thread

import supervisely as sly
from supervisely.io.fs import silent_remove


class PersistentImageLRUCache(LRUCache):
    __marker = object()

    def __init__(self, maxsize, filepath: Path, getsizeof=None):
        super().__init__(maxsize)
        self._base_dir = filepath

    def __getitem__(self, key: Any) -> Any:
        filepath = super(PersistentImageLRUCache, self).__getitem__(key)
        return sly.image.read(str(filepath))

    def __setitem__(self, key: Any, value: Any) -> None:
        if not self._base_dir.exists():
            self._base_dir.mkdir()

        filepath = self._base_dir / f"{str(key)}.png"
        super(PersistentImageLRUCache, self).__setitem__(key, filepath)

        if filepath.exists():
            sly.logger.debug(f"Rewrite image {str(filepath)}")
        sly.image.write(str(filepath), value)

    def pop(self, key, default=__marker):
        if key in self:
            filepath = self._base_dir / f"{str(key)}.png"
            value = self[key]
            del self[key]
            silent_remove(filepath)
            sly.logger.debug(f"Remove {filepath} frame")
        elif default is self.__marker:
            raise KeyError(key)
        else:
            value = default
        return value

    def clear(self, rm_base_folder=True) -> None:
        while self.currsize > 0:
            self.popitem()
        if rm_base_folder:
            shutil.rmtree(self._base_dir)


class PersistentImageTTLCache(TTLCache):
    def __init__(self, maxsize: int, ttl: int, filepath: Path):
        super().__init__(maxsize, ttl)
        self._base_dir = filepath

    def __getitem__(self, key: Any) -> np.ndarray:
        filepath = super(PersistentImageTTLCache, self).__getitem__(key)
        return sly.image.read(str(filepath))

    def __setitem__(self, key: Any, value: np.ndarray) -> None:
        if not self._base_dir.exists():
            self._base_dir.mkdir()

        filepath = self._base_dir / f"{str(key)}.png"
        super(PersistentImageTTLCache, self).__setitem__(key, filepath)

        if filepath.exists():
            sly.logger.debug(f"Rewrite image {str(filepath)}")
        sly.image.write(str(filepath), value)

    def __delitem__(self, key: Any) -> None:
        self.__del_file(key)
        return super().__delitem__(key)

    def __del_file(self, key: Any):
        filepath = self._base_dir / f"{str(key)}.png"
        silent_remove(filepath)

    def __get_keys(self):
        return self._TTLCache__links.keys()

    def expire(self, time=None):
        existing = set(self.__get_keys())
        super().expire(time)
        deleted = existing.difference(self.__get_keys())
        sly.logger.debug(f"Deleted keys: {deleted}")

        for key in deleted:
            self.__del_file(key)

    def clear(self, rm_base_folder=True) -> None:
        while self.currsize > 0:
            self.popitem()
        if rm_base_folder:
            shutil.rmtree(self._base_dir)


class InferenceImageCache:
    class _LoadType(Enum):
        ImageId: str = "IMAGE"
        ImageHash: str = "HASH"
        Frame: str = "FRAME"

    def __init__(
        self,
        maxsize: int,
        ttl: int,
        is_persistent: bool = True,
        base_folder: str = sly.env.smart_cache_container_dir(),
    ) -> None:
        self._is_persistent = is_persistent
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = Lock()
        self._load_queue = CacheOut(ttl=5)

        if is_persistent:
            self._data_dir = Path(base_folder)
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._cache = PersistentImageTTLCache(maxsize, ttl, self._data_dir)
        else:
            self._cache = TTLCache(maxsize, ttl)

    def clear_cache(self):
        with self._lock:
            self._cache.clear(False)

    def download_image(self, api: sly.Api, image_id: int):
        name = self._image_name(image_id)
        self._wait_if_in_queue(name, api.logger)

        if name not in self._cache:
            self._load_queue.set(name, image_id)
            api.logger.debug(f"Add image #{image_id} to cache")
            img = api.image.download_np(image_id)
            self._add_to_cache(name, img)
            return img

        api.logger.debug(f"Get image #{image_id} from cache")
        return self._cache[name]

    def download_images(self, api: sly.Api, dataset_id: int, image_ids: List[int], **kwargs):
        return_images = kwargs.get("return_images", True)

        def load_generator(image_ids: List[int]):
            return api.image.download_nps_generator(dataset_id, image_ids)

        return self._download_many(
            image_ids,
            self._image_name,
            load_generator,
            api.logger,
            return_images,
        )

    def download_image_by_hash(self, api: sly.Api, img_hash: str) -> np.ndarray:
        image_key = self._image_name(img_hash)
        self._wait_if_in_queue(image_key, api.logger)

        if image_key not in self._cache:
            self._load_queue.set(image_key, img_hash)
            image = api.image.download_nps_by_hashes([img_hash])
            self._add_to_cache(image_key, image)
            return image
        return self._cache[image_key]

    def download_images_by_hashes(
        self, api: sly.Api, img_hashes: List[str], **kwargs
    ) -> List[np.ndarray]:
        return_images = kwargs.get("return_images", True)

        def load_generator(img_hashes: List[str]):
            return api.image.download_nps_by_hashes_generator(img_hashes)

        return self._download_many(
            img_hashes,
            self._image_name,
            load_generator,
            api.logger,
            return_images,
        )

    def download_frame(self, api: sly.Api, video_id: int, frame_index: int) -> np.ndarray:
        name = self._frame_name(video_id, frame_index)
        self._wait_if_in_queue(name, api.logger)

        if name not in self._cache:
            self._load_queue.set(name, (video_id, frame_index))
            frame = api.video.frame.download_np(video_id, frame_index)
            self._add_to_cache(name, frame)
            api.logger.debug(f"Add frame #{frame_index} for video #{video_id} to cache")
            return frame

        api.logger.debug(f"Get frame #{frame_index} for video #{video_id} from cache")
        return self._cache[name]

    def download_frames(
        self, api: sly.Api, video_id: int, frame_indexes: List[int], **kwargs
    ) -> List[np.ndarray]:
        return_images = kwargs.get("return_images", True)

        def name_constuctor(frame_index: int):
            return self._frame_name(video_id, frame_index)

        def load_generator(frame_indexes: List[int]):
            return api.video.frame.download_nps_generator(video_id, frame_indexes)

        return self._download_many(
            frame_indexes,
            name_constuctor,
            load_generator,
            api.logger,
            return_images,
        )

    def add_cache_endpoint(self, server: FastAPI):
        @server.post("/smart_cache")
        def cache_endpoint(request: Request):
            return self.cache_task(
                api=request.state.api,
                state=request.state.state,
            )

    def cache_task(self, api: sly.Api, state: dict):
        api.logger.debug("Request state in cache endpoint", extra=state)
        image_ids, task_type = self._parse_state(state)
        kwargs = {"return_images": False}

        if task_type is InferenceImageCache._LoadType.ImageId:
            if "dataset_id" in state:
                self.download_images(api, state["dataset_id"], image_ids, **kwargs)
            else:
                for img_id in image_ids:
                    self.download_image(api, img_id)
        elif task_type is InferenceImageCache._LoadType.ImageHash:
            self.download_images_by_hashes(api, image_ids, **kwargs)
        elif task_type is InferenceImageCache._LoadType.Frame:
            video_id = state["video_id"]
            self.download_frames(api, video_id, image_ids, **kwargs)

    def run_cache_task_manually(
        self,
        api: sly.Api,
        list_of_ids_ranges_or_hashes: List[Union[str, int, List[int]]],
        *,
        dataset_id: Optional[int] = None,
        video_id: Optional[int] = None,
    ) -> None:
        """
        Run cache_task in new thread.

        :param api: supervisely api
        :type api: sly.Api
        :param list_of_ids_ranges_or_hashes: information abount images/frames need to be loaded;
        to download images, pass list of integer IDs (`dataset_id` requires)
        or list of hash strings (`dataset_id` could be None);
        to download frames, pass list of pairs of indices of the first and last frame
        and `video_id` (ex.: [[1, 3], [5, 5], [7, 10]])
        :type list_of_ids_ranges_or_hashes: List[Union[str, int, List[int]]]
        :param dataset_id: id of dataset on supervisely platform; default is None
        :type dataset_id: Optional[int]
        :param video_id: id of video on supervisely platform; default is None
        :type video_id: Optional[int]
        """
        state = {}
        if isinstance(list_of_ids_ranges_or_hashes[0], str):
            api.logger.debug("Got a task to add images using hash")
            state["image_hashes"] = list_of_ids_ranges_or_hashes
        elif video_id is None:
            if dataset_id is None:
                api.logger.error("dataset_id or video_id must be defined if not hashes are used")
                return
            api.logger.debug("Got a task to add images using IDs")
            state["image_ids"] = list_of_ids_ranges_or_hashes
            state["dataset_id"] = dataset_id
        else:
            api.logger.debug("Got a task to add frames")
            state["video_id"] = video_id
            state["frame_ranges"] = list_of_ids_ranges_or_hashes

        thread = Thread(target=self.cache_task, kwargs={"api": api, "state": state})
        thread.start()

    @property
    def ttl(self):
        return self._ttl

    @property
    def tmp_path(self):
        if self._is_persistent:
            return str(self._data_dir)
        return None

    def _parse_state(self, state: dict) -> Tuple[List[Any], _LoadType]:
        if "image_ids" in state:
            return state["image_ids"], InferenceImageCache._LoadType.ImageId
        elif "image_hashes" in state:
            return state["image_hashes"], InferenceImageCache._LoadType.ImageHash
        elif "video_id" in state:
            frame_ranges = state["frame_ranges"]
            frames = []
            for fr_range in frame_ranges:
                shift = 1
                if fr_range[0] > fr_range[1]:
                    shift = -1
                start, end = fr_range[0], fr_range[1] + shift
                frames.extend(list(range(start, end, shift)))
            return frames, InferenceImageCache._LoadType.Frame
        raise ValueError("State has no proper fields: image_ids, image_hashes or video_id")

    def _add_to_cache(
        self,
        names: Union[str, List[str]],
        images: Union[np.ndarray, List[np.ndarray]],
    ):
        if isinstance(names, str):
            names = [names]

        if isinstance(images, np.ndarray):
            images = [images]

        if len(images) != len(names):
            raise ValueError(
                f"Number of images and names do not match: {len(images)} != {len(names)}"
            )

        for name, img in zip(names, images):
            with self._lock:
                self._cache[name] = img
                self._load_queue.delete(name)

    def _image_name(self, id_or_hash: Union[str, int]) -> str:
        if isinstance(id_or_hash, int):
            return f"image_{id_or_hash}"
        hash_wo_slash = id_or_hash.replace("/", "-")
        return f"image_{hash_wo_slash}"

    def _frame_name(self, video_id: int, frame_index: int) -> str:
        return f"frame_{video_id}_{frame_index}"

    def _download_many(
        self,
        indexes: List[Union[int, str]],
        name_cunstructor: Callable[[int], str],
        load_generator: Callable[
            [List[int]],
            Generator[Tuple[Union[int, str], np.ndarray], None, None],
        ],
        logger: Logger,
        return_images: bool = True,
    ) -> Optional[List[np.ndarray]]:
        indexes_to_load = []
        pos_by_name = {}
        all_frames = [None for _ in range(len(indexes))]

        for pos, hash_or_id in enumerate(indexes):
            name = name_cunstructor(hash_or_id)
            self._wait_if_in_queue(name, logger)

            if name not in self._cache:
                self._load_queue.set(name, hash_or_id)
                indexes_to_load.append(hash_or_id)
                pos_by_name[name] = pos
            elif return_images is True:
                all_frames[pos] = self._cache[name]

        if len(indexes_to_load) > 0:
            for id_or_hash, image in load_generator(indexes_to_load):
                name = name_cunstructor(id_or_hash)
                self._add_to_cache(name, image)

                if return_images:
                    pos = pos_by_name[name]
                    all_frames[pos] = image

        logger.debug(f"All stored files: {sorted(os.listdir(self.tmp_path))}")
        logger.debug(f"Images/Frames added to cache: {indexes_to_load}")
        logger.debug(f"Images/Frames founded in cache: {set(indexes).difference(indexes_to_load)}")

        if return_images:
            return all_frames
        return

    def _wait_if_in_queue(self, name, logger: Logger):
        if name in self._load_queue:
            logger.debug(f"Waiting for other task to load {name}")

        while name in self._load_queue:
            # TODO: sleep if slowdown
            sleep(0.1)
            continue
