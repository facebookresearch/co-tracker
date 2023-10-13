# coding: utf-8

import os

# Do NOT use directly for video extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_VIDEO_EXTENSIONS = ['.avi', '.mkv', '.mp4']


class VideoExtensionError(Exception):
    pass


def is_valid_ext(ext: str) -> bool:
    '''
    The function is_valid_ext checks file extension for list of supported video extensions('.avi', '.mp4')
    :param ext: file extention
    :return: True if file extention in list of supported images extensions, False - in otherwise
    '''
    return ext.lower() in ALLOWED_VIDEO_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    '''
    The function has_valid_ext checks if a given file has a supported extension('.avi', '.mp4')
    :param path: the path to the input file
    :return: True if a given file has a supported extension, False - in otherwise
    '''
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str):
    '''
    The function validate_ext generate exception error if file extention is not in list of supported videos
    extensions('.avi', '.mp4')
    :param ext: file extention
    '''
    if not is_valid_ext(ext):
        raise VideoExtensionError('Unsupported video extension: {}. Only the following extensions are supported: {}.'
                                  .format(ALLOWED_VIDEO_EXTENSIONS))

