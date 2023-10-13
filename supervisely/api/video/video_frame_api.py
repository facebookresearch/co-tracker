# coding: utf-8

# docs
from __future__ import annotations

import re
from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
from requests_toolbelt import MultipartDecoder
from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.module_api import ApiField, ModuleApi
from supervisely.imaging import image as sly_image
from supervisely.io.fs import ensure_base_path


class VideoFrameAPI(ModuleApi):
    """
    :class:`Frame<supervisely.video_annotation.frame.Frame>` for a single video. :class:`VideoFrameAPI<VideoFrameAPI>` object is immutable.
    """

    def _download(self, video_id: int, frame_index: int):
        """
        Private method. Download frame with given video ID and frame index.

        :param video_id: int
        :param frame_index: int
        :return: Response class object containing frame data with given index from given video id
        """

        response = self._api.post(
            "videos.download-frame", {ApiField.VIDEO_ID: video_id, ApiField.FRAME: frame_index}
        )
        return response

    def _download_batch(
        self,
        video_id: int,
        frame_indexes: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ):
        """
        Private method. Batch download frames with given video ID and frame indexes.

        :param video_id: int
        :param frame_indexes: List[int]
        :return: Response class object containing frame data with given index from given video id
        """

        for batch_ids in batched(frame_indexes):
            response = self._api.post(
                "videos.bulk.download-frame",
                {ApiField.VIDEO_ID: video_id, ApiField.FRAMES: batch_ids},
            )
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                frame_id = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])

                if progress_cb is not None:
                    progress_cb(1)
                yield frame_id, part

    def download_np(self, video_id: int, frame_index: int) -> np.ndarray:
        """
        Download Image for frame with given index from given video ID in numpy format (RGB).

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_index: Index of frame to download.
        :type frame_index: int
        :return: Image in RGB numpy matrix format
        :rtype: :class:`np.ndarray`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198703211
            frame_idx = 5
            image_np = api.video.frame.download_np(video_id, frame_idx)
        """

        response = self._download(video_id, frame_index)
        frame = sly_image.read_bytes(response.content)
        return frame

    def download_nps(
        self,
        video_id: int,
        frame_indexes: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        keep_alpha: Optional[bool] = False,
    ) -> List[np.ndarray]:
        """
        Download frames with given indexes from given video ID in numpy format(RGB).

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_indexes: Indexes of frames to download.
        :type frame_indexes: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: List of Images in RGB numpy matrix format
        :rtype: List[np.ndarray]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198703211
            frame_indexes = [1,2,3,4,5,10,11,12,13,14,15]
            images_np = api.video.frame.download_nps(video_id, frame_indexes)
        """

        downloaded_frames = []
        for frame_bytes, frame_idx in zip(
            self.download_bytes(
                video_id=video_id, frame_indexes=frame_indexes, progress_cb=progress_cb
            ),
            frame_indexes,
        ):
            try:
                frame = sly_image.read_bytes(frame_bytes, keep_alpha)
                downloaded_frames.append(frame)
            except Exception as e:
                raise Exception(f"Couldn't read frame: {frame_idx}.") from e
        return downloaded_frames

    def download_nps_generator(
        self,
        video_id: int,
        frame_indexes: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        keep_alpha: Optional[bool] = False,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        for frame_idx, resp_part in self._download_batch(video_id, frame_indexes, progress_cb):
            frame_bytes = resp_part.content
            try:
                yield frame_idx, sly_image.read_bytes(frame_bytes, keep_alpha)
            except Exception as e:
                raise Exception(f"Couldn't read frame: {frame_idx}.") from e

    def download_path(self, video_id: int, frame_index: int, path: str) -> None:
        """
        Downloads frame on the given path for frame with given index from given Video ID.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_index: Index of frame to download.
        :type frame_index: int
        :param path: Local save path for Image.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198703211
            frame_idx = 5
            save_path = '/home/admin/Downloads/frames/result.png'
            api.video.frame.download_path(video_id, frame_idx, save_path)
        """

        response = self._download(video_id, frame_index)
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def download_paths(
        self,
        video_id: int,
        frame_indexes: List[int],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Downloads frames to given paths for frames with given indexes from given Video ID.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_indexes: Indexes of frames to download.
        :type frame_indexes: List[int]
        :param paths: Local save paths for frames.
        :type paths: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198703211
            frame_indexes = [1,2,3,4,5,10,11,12,13,14,15]
            save_paths = [f"/home/admin/projects/video_project/frames/{idx}.png" for idx in frame_indexes]
            api.video.frame.download_paths(video_id, frame_indexes, save_paths)
        """

        if len(frame_indexes) == 0:
            return
        if len(frame_indexes) != len(paths):
            raise ValueError(
                'Can not match "indexes" and "paths" lists, len(frame_indexes) != len(paths)'
            )

        idx_to_path = {idx: path for idx, path in zip(frame_indexes, paths)}
        for frame_id, resp_part in self._download_batch(video_id, frame_indexes, progress_cb):
            with open(idx_to_path[frame_id], "wb") as w:
                w.write(resp_part.content)

    def download_bytes(
        self,
        video_id: int,
        frame_indexes: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[bytes]:
        """
        Download frames with given indexes from Dataset in Binary format.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_indexes: List of video frames indexes in Supervisely.
        :type frame_indexes: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: List of Images in binary format
        :rtype: :class:`List[bytes]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 213542
            frame_indexes = [1,2,3,4,5,10,11,12,13,14,15]
            frames_bytes = api.video.frame.download_bytes(video_id=video_id, frame_indexes=frame_indexes)
            print(frames_bytes)
            # Output: [b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\...]
        """

        if len(frame_indexes) == 0:
            return []

        idx_to_frame = {}
        for frame_idx, resp_part in self._download_batch(video_id, frame_indexes, progress_cb):
            idx_to_frame[frame_idx] = resp_part.content
        return [idx_to_frame[idx] for idx in frame_indexes]
