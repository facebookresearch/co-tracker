# coding: utf-8
from __future__ import annotations

import random
import colorsys
import os
import gzip
import json
import copy
from typing import List


def _validate_color(color):
    """
    Checks input color for compliance with the required format
    :param: color: color (RGB tuple of integers)
    """
    if not isinstance(color, (list, tuple)):
        raise ValueError("Color has to be list, or tuple")
    if len(color) != 3:
        raise ValueError("Color have to contain exactly 3 values: [R, G, B]")
    for channel in color:
        validate_channel_value(channel)


def random_rgb() -> List[int, int, int]:
    """
    Generate RGB color with fixed saturation and lightness.

    :return: RGB integer values
    :rtype: :class:`List[int, int, int]`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        color = sly.color.random_rgb()
        print(color)
        # Output: [138, 15, 123]
    """
    hsl_color = (random.random(), 0.3, 0.8)
    rgb_color = colorsys.hls_to_rgb(*hsl_color)
    return [round(c * 255) for c in rgb_color]


def _normalize_color(color):
    """
    Divide all RGB values by 255.
    :param color: color (RGB tuple of integers)
    """
    return [c / 255.0 for c in color]


def _color_distance(first_color: list, second_color: list) -> float:
    """
    Calculate distance in HLS color space between Hue components of 2 colors
    :param first_color: first color (RGB tuple of integers)
    :param second_color: second color (RGB tuple of integers)
    :return: Euclidean distance between 'first_color' and 'second_color'
    """
    first_color_hls = colorsys.rgb_to_hls(*_normalize_color(first_color))
    second_color_hls = colorsys.rgb_to_hls(*_normalize_color(second_color))
    hue_distance = min(
        abs(first_color_hls[0] - second_color_hls[0]),
        1 - abs(first_color_hls[0] - second_color_hls[0]),
    )
    return hue_distance


def generate_rgb(exist_colors: List[List[int, int, int]]) -> List[int, int, int]:
    """
    Generate new color which oppositely by exist colors.

    :param exist_colors: List of existing colors in RGB format.
    :type exist_colors: list
    :return: RGB integer values
    :rtype: :class:`List[int, int, int]`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        exist_colors = [[0, 0, 0], [128, 64, 255]]
        color = sly.color.generate_rgb(exist_colors)
        print(color)
        # Output: [15, 138, 39]
    """
    largest_min_distance = 0
    best_color = random_rgb()
    if len(exist_colors) > 0:
        for _ in range(100):
            color = random_rgb()
            current_min_distance = min(_color_distance(color, c) for c in exist_colors)
            if current_min_distance > largest_min_distance:
                largest_min_distance = current_min_distance
                best_color = color
    _validate_color(best_color)
    return best_color


def rgb2hex(color: List[int, int, int]) -> str:
    """
    Convert integer color format to HEX string.

    :param color: List of existing colors in RGB format.
    :type color: List[int, int, int]
    :return: HEX RGB string
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        hex_color = sly.color.rgb2hex([128, 64, 255])
        print(hex_color)
        # Output: #8040FF
    """
    _validate_color(color)
    return "#" + "".join("{:02X}".format(component) for component in color)


def _hex2color(hex_value: str) -> list:
    """
    Convert HEX RGB string to integer RGB format
    :param hex_value: HEX RGBA string. Example: "#FF02А4
    :return: RGB integer values. Example: [80, 255, 0]
    """
    assert hex_value.startswith("#")
    return [int(hex_value[i : (i + 2)], 16) for i in range(1, len(hex_value), 2)]


def hex2rgb(hex_value: str) -> List[int, int, int]:
    """
    Convert HEX RGB string to integer RGB format.

    :param hex_value: HEX RGB string.
    :type hex_value: str
    :return: RGB integer values
    :rtype: :class:`List[int, int, int]`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        hex_color = '#8040FF'
        color = sly.color.hex2rgb(hex_color)
        print(color)
        # Output: [128, 64, 255]
    """
    assert len(hex_value) == 7, "Supported only HEX RGB string format!"
    color = _hex2color(hex_value)
    _validate_color(color)
    return color


def _hex2rgba(hex_value: str) -> list:
    """
    Convert HEX RGBA string to integer RGBA format
    :param hex_value: HEX RGBA string. Example: "#FF02А4CC
    :return: RGBA integer values. Example: [80, 255, 0, 128]
    """
    assert len(hex_value) == 9, "Supported only HEX RGBA string format!"
    return _hex2color(hex_value)


def validate_channel_value(value: int) -> None:
    """
    Generates ValueError if value not between 0 and 255.

    :param value: Input channel value.
    :type value: int
    :raises: :class:`ValueError` if value not between 0 and 255.
    :return: None
    :rtype: :class:`NoneType`
    """
    if 0 <= value <= 255:
        pass
    else:
        raise ValueError("Color channel has to be in range [0; 255]")


# generate colors
# import distinctipy
# data = {}
# for n in range(100):
#     print(n)
#     colors = distinctipy.get_colors(n)
#     rgb_colors = [distinctipy.get_rgb256(color) for color in colors]
#     data[n] = rgb_colors
# sly.json.dump_json_file(data, "colors.json")

# import gzip
# data = sly.json.load_json_file("colors.json")
# with gzip.open("colors.json.gz", "wt", encoding="UTF-8") as zipfile:
#     json.dump(data, zipfile)
# with gzip.open("colors.json.gz", "r") as fin:
#     data = json.loads(fin.read().decode("utf-8"))


def get_predefined_colors(n: int):
    try:
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colors.json.gz")
        with gzip.open(file, "r") as fin:
            data = json.loads(fin.read().decode("utf-8"))

        if str(n) in data:
            colors = copy.deepcopy(data[str(n)])
            random.Random(7).shuffle(colors)
            return colors
        # generate random colors
    except Exception as e:
        print(repr(e))

    rand_colors = []
    for i in range(n):
        rand_colors.append(random_rgb())
    return rand_colors
