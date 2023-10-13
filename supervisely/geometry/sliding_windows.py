# coding: utf-8

from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.validation import is_2d_int_coords_valid


class SlidingWindows:
    """
    """
    def __init__(self, window_shape, min_overlap):
        if not is_2d_int_coords_valid([window_shape]):
            raise ValueError('window_shape must contains 2 integers.')

        if not is_2d_int_coords_valid([min_overlap]):
            raise ValueError('min_overlap must contains 2 integers.')

        self.window_shape = tuple(window_shape)
        self.min_overlap = tuple(min_overlap)
        self.stride = tuple(self.window_shape[i] - self.min_overlap[i] for i in (0, 1))
        if min(self.stride) < 1:
            raise RuntimeError('Wrong sliding window settings, overlap is too high.')

    def get(self, source_shape):
        """
        """
        source_rect = Rectangle.from_size(source_shape)
        window_rect = Rectangle.from_size(self.window_shape)
        if not source_rect.contains(window_rect):
            raise RuntimeError('Sliding window: window is larger than source (image).')

        hw_limit = tuple(source_shape[i] - self.window_shape[i] for i in (0, 1))
        for wind_top in range(0, hw_limit[0] + self.stride[0], self.stride[0]):
            for wind_left in range(0, hw_limit[1] + self.stride[1], self.stride[1]):
                window_top = min(wind_top, hw_limit[0])
                window_left = min(wind_left, hw_limit[1])
                roi = window_rect.translate(drow=window_top, dcol=window_left)
                if not source_rect.contains(roi):
                    raise RuntimeError('Sliding window: result crop bounds are invalid.')
                yield roi
