# coding: utf-8
"""Functions for processing videos"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional

import os
from supervisely import logger as default_logger
from supervisely.io.fs import get_file_name, get_file_ext
from supervisely._utils import rand_str, is_development, abs_url


# Do NOT use directly for video extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_VIDEO_EXTENSIONS = [".avi", ".mp4", ".3gp", ".flv", ".webm", ".wmv", ".mov", ".mkv"]


_SUPPORTED_CONTAINERS = {"mp4", "webm", "ogg", "ogv"}
_SUPPORTED_CODECS = {"h264", "vp8", "vp9"}


class VideoExtensionError(Exception):
    pass


class UnsupportedVideoFormat(Exception):
    pass


class VideoReadException(Exception):
    pass


def is_valid_ext(ext: str) -> bool:
    """
    Checks if given extension is supported.

    :param ext: Video file extension.
    :type ext: str
    :return: bool
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        sly.video.is_valid_ext(".mp4")  # True
        sly.video.is_valid_ext(".jpeg") # False
    """
    return ext.lower() in ALLOWED_VIDEO_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    """
    Checks if Video file from given path has supported extension.

    :param path: Path to Video file.
    :type path: str
    :return: bool
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        video_path = "/home/admin/work/videos/Cars/ds0/video/6x.mp4"
        sly.video.has_valid_ext(video_path) # True
    """
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str):
    """
    Raises error if given extension is not supported.

    :param ext: Video extension. Available extensions: avi, mp4, 3gp, flv, webm, wmv, mov, mkv.
    :type ext: str
    :raises: :class:`UnsupportedVideoFormat` if given video with extension that is not supported.
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

       import supervisely as sly

        sly.video.validate_ext(".jpeg")
        # Unsupported video extension: .jpeg.
        # Only the following extensions are supported: ['.avi', '.mp4', '.3gp', '.flv', '.webm', '.wmv', '.mov', '.mkv'].
    """
    if not is_valid_ext(ext):
        raise UnsupportedVideoFormat(
            "Unsupported video extension: {}. Only the following extensions are supported: {}.".format(
                ext, ALLOWED_VIDEO_EXTENSIONS
            )
        )


def get_image_size_and_frames_count(path: str) -> Tuple[Tuple[int, int], int]:
    """
    Gets image size and number of frames from Video file.

    :param path: Path to Video file.
    :type path: str
    :return: Image size and number of Video frames.
    :rtype: :class:`Tuple[Tuple[int, int], int]`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        video_path = "/home/admin/work/videos/Cars/ds0/video/6x.mp4"
        video_info = sly.video.get_image_size_and_frames_count(video_path)
        print(video_info)
        # Output: ((720, 1280), 152)
    """
    import skvideo.io

    vreader = skvideo.io.FFmpegReader(path)
    vlength = vreader.getShape()[0]
    img_height = vreader.getShape()[1]
    img_width = vreader.getShape()[2]

    img_size = (img_height, img_width)

    return img_size, vlength


def validate_format(path: str) -> None:
    """
    Raise error if Video file from given path couldn't be read or file extension is not supported.

    :param path: Path to Video file.
    :type path: str
    :raises: :class:`VideoReadException` if Video file from given path couldn't be read or file extension is not supported
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        video_path = "/home/paul/work/sphinx-docs/supervisely_py/docs/source/debug/video/Prius_360/ds0/video/video.jpg"
        sly.video.validate_format(video_path)
        # Unsupported video extension: .jpg. Only the following extensions are supported: ['.avi', '.mp4', '.3gp', '.flv', '.webm', '.wmv', '.mov', '.mkv'].
    """
    try:
        get_image_size_and_frames_count(path)
    except Exception as e:
        raise VideoReadException(
            "Error has occured trying to read video {!r}. Original exception message: {!r}".format(
                path, str(e)
            )
        )

    validate_ext(os.path.splitext(path)[1])


def _check_video_requires_processing(video_info, stream_info):
    """
    Check if video need container or codec processing
    :param video_info: dict
    :param stream_info: dict
    :return: bool
    """
    need_process_container = True
    for name in video_info["meta"]["formatName"].split(","):
        name = name.strip().split(".")[-1]
        if name in _SUPPORTED_CONTAINERS:
            need_process_container = False
            break

    need_process_codec = True
    codec = stream_info["codecName"]
    if codec in _SUPPORTED_CODECS:
        need_process_codec = False

    if (need_process_container is False) and (need_process_codec is False):
        return False

    return True


def count_video_streams(all_streams: List[Dict]) -> int:
    """
    Count number of video streams in video.

    :param all_streams: List of Video file audio and video streams.
    :type all_streams: List[dict]
    :return: Number of video streams in Video file
    :rtype: :class:`int`
    """
    count = 0
    for stream_info in all_streams:
        if stream_info["codecType"] == "video":
            count += 1
    return count


def get_video_streams(all_streams: List[Dict]) -> List:
    """
    Get list of video streams from given list of all streams.

    :param all_streams: List of Video file audio and video streams.
    :type all_streams: List[dict]
    :return: List of video streams in Video file.
    :rtype: :class:`list`
    """
    video_streams = []
    for stream_info in all_streams:
        if stream_info["codecType"] == "video":
            video_streams.append(stream_info)
    return video_streams


def warn_video_requires_processing(file_name: str, logger: Optional[default_logger] = None) -> None:
    """
    Create logger if it was not there and displays message about the need for transcoding.

    :param file_name: Video file name.
    :type file_name: str
    :param logger: Logger object.
    :type logger: logger
    :return: None
    :rtype: :class:`NoneType`
    """
    if logger is None:
        logger = default_logger
    logger.warning(
        "Video Stream {!r} is skipped: requires transcoding. Transcoding is supported only in Enterprise Edition (EE)".format(
            file_name
        )
    )


def gen_video_stream_name(file_name: str, stream_index: int) -> str:
    """
    Create name to video stream from given filename and index of stream.

    :param file_name: Video file name.
    :type file_name: str
    :param stream_index: Stream index.
    :type stream_index: int
    :return: str
    :rtype: str
    :Usage example:

     .. code-block:: python

        stream_name = gen_video_stream_name('my_video.mp4', 2)
        print(stream_name)
        # Output: my_video_stream_2_CULxO.mp4
    """
    return "{}_stream_{}_{}{}".format(
        get_file_name(file_name), stream_index, rand_str(5), get_file_ext(file_name)
    )


def get_info(video_path: str, cpu_count: Optional[int] = None) -> Dict:
    """
    Get information about video from given path.

    :param video_path: Video file path.
    :type video_path: str
    :param cpu_count: CPU count.
    :type cpu_count: int
    :raises: :class:`ValueError` if no video streams found.
    :return: Information about video
    :rtype: :class:`Dict`
    :Usage example:

     .. code-block:: python

        from supervisely.video.video import get_info
        video_info = get_info('/home/video/1.mp4')
        print(json.dumps(video_info, indent=4))
        # Output: {
        #     "streams": [
        #         {
        #             "index": 0,
        #             "width": 1920,
        #             "height": 1080,
        #             "duration": 16.666667,
        #             "rotation": 0,
        #             "codecName": "mpeg4",
        #             "codecType": "video",
        #             "startTime": 0,
        #             "framesCount": 500,
        #             "framesToTimecodes": [
        #                 0.0,
        #                 0.033333,
        #                 0.066667,
        #                 0.1,
        #                   ...
        #                 16.566667,
        #                 16.6,
        #                 16.633333
        #             ]
        #         }
        #     ],
        #     "formatName": "mov,mp4,m4a,3gp,3g2,mj2",
        #     "duration": 16.667,
        #     "size": "61572600"
        # }
    """
    import pathlib
    import subprocess, os, ast, math
    from subprocess import PIPE

    def rotate_dimensions(width, height, rotation):
        cur_angle = rotation * math.pi / 180
        c = math.cos(cur_angle)
        s = math.sin(cur_angle)
        w = round(abs(width * c - height * s))
        h = round(abs(width * s + height * c))
        return w, h

    if cpu_count is None:
        cpu_count = os.cpu_count()

    session = subprocess.Popen(
        [
            "ffprobe",
            "-i",
            f"{video_path}",
            "-threads",
            f"{cpu_count}",
            "-fflags",
            "+genpts",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "frame=stream_index,pkt_pts_time",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = session.communicate()
    if len(stderr) != 0:
        default_logger.warning(stderr.decode("utf-8"))
        # * Return error instead of warning if some problems will appear.
        # raise RuntimeError(stderr.decode("utf-8"))

    video_meta = ast.literal_eval(stdout.decode("utf-8"))

    frames_to_timecodes = []
    has_video = False
    # ? Assigned but never used, consider removing.
    # audio_stream_info = None

    for frame in video_meta["frames"]:
        if frame["stream_index"] == 0:
            frames_to_timecodes.append(float(frame.get("pkt_pts_time", frame.get("pts_time"))))

    stream_infos = []
    for stream in video_meta["streams"]:
        duration = stream.get("duration", video_meta["format"]["duration"])
        if stream["codec_type"] == "video":
            has_video = True
            stream_info = {
                "index": stream["index"],
                "width": stream["width"],
                "height": stream["height"],
                "duration": float(duration),
                "rotation": 0,
                "codecName": stream["codec_name"],
                "codecType": stream["codec_type"],
                "startTime": int(float(stream["start_time"])),
                "framesCount": len(frames_to_timecodes),
                "framesToTimecodes": frames_to_timecodes,
            }
            side_data_list = stream.get("side_data_list", None)
            if side_data_list:
                for data in side_data_list:
                    rotation = data.get("rotation", None)
                if rotation:
                    stream_info["rotation"] = rotation
                    width, height = rotate_dimensions(
                        stream_info["width"], stream_info["height"], rotation
                    )
                    stream_info["originalWidth"] = stream_info["width"]
                    stream_info["originalHeight"] = stream_info["height"]
                    stream_info["width"] = width
                    stream_info["height"] = height
        elif stream["codec_type"] == "audio":
            stream_info = {
                "index": stream["index"],
                "channels": stream["channels"],
                "duration": float(duration),
                "codecName": stream["codec_name"],
                "codecType": stream["codec_type"],
                "startTime": int(float(stream["start_time"])),
                "sampleRate": int(stream["sample_rate"]),
            }
        else:
            continue
        stream_infos.append(stream_info)

    if has_video is False:
        raise ValueError("No video streams found")

    result = {
        "streams": stream_infos,
        "formatName": video_meta["format"]["format_name"],
        "duration": float(video_meta["format"]["duration"]),
        "size": video_meta["format"]["size"],
    }

    return result


def get_labeling_tool_url(dataset_id, video_id):
    res = f"/app/videos_v2/?datasetId={dataset_id}&videoId={video_id}&videoFrame=0"
    if is_development():
        res = abs_url(res)
    return res


def get_labeling_tool_link(url, name="open in labeling tool"):
    return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{name}<i class="zmdi zmdi-open-in-new" style="margin-left: 5px"></i></a>'
