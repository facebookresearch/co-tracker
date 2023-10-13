# coding: utf-8

# docs
from typing import Tuple, Optional

import os

from PIL import ImageFont
from supervisely.io.fs import get_file_ext, file_exists

FONT_EXTENSION = ".ttf"
DEFAULT_FONT_FILE_NAME = "DejaVuSansMono.ttf"
_fonts = {}  # (font name, size) -> font


def _get_font_path_by_name(font_file_name: str) -> str:
    """
    Walk over systems fonts paths and match with given font file name.

    :param font_file_name: Nameame of the font file.
    :type font_file_name: str
    :return: Full path of requested by name font or None if file not found in system paths.
    :rtype: str
    """
    import matplotlib.font_manager as fontman

    fonts_paths_list = fontman.findSystemFonts()
    for path in fonts_paths_list:
        if os.path.basename(path) == font_file_name:
            return path
    return None


def load_font(
        font_file_name: str, font_size: Optional[int] = 12
) -> ImageFont.FreeTypeFont:
    """
    Set global font true-type for drawing.

    :param font_file_name: name of font file (example: 'DejaVuSansMono.ttf')
    :type font_file_name: str
    :param font_size: selected font size
    :type font_size: int
    :return: Font object
    :rtype: PIL.ImageFont.FreeTypeFont
    """
    if get_file_ext(font_file_name) == FONT_EXTENSION:
        font_path = _get_font_path_by_name(font_file_name)
        if (font_path is not None) and file_exists(font_path):
            return ImageFont.truetype(font_path, font_size, encoding="utf-8")
        else:
            raise ValueError(
                'Font file "{}" not found in system paths. Try to set another font.'.format(
                    font_file_name
                )
            )
    else:
        raise ValueError("Supported only TrueType fonts!")


def get_font(
        font_file_name: Optional[str] = None, font_size: Optional[int] = 12
) -> ImageFont.FreeTypeFont:
    """
    Args:
        font_file_name: name of font file (example: 'DejaVuSansMono.ttf')
        font_size: selected font size
    Returns:
        font for drawing
    """
    if font_file_name is None:
        font_file_name = DEFAULT_FONT_FILE_NAME

    font_key = (font_file_name, font_size)
    if font_key not in _fonts:
        _fonts[font_key] = load_font(font_file_name, font_size)
    return _fonts[font_key]


def get_readable_font_size(img_size: Tuple[int, int]) -> int:
    """
    Get size of font for image with given sizes
    :param img_size: size of image
    :return: size of font
    """
    minimal_font_size = 6
    base_font_size = 14
    base_image_size = 512
    return max(
        minimal_font_size,
        round(base_font_size * (img_size[0] + img_size[1]) // 2) // base_image_size,
    )
