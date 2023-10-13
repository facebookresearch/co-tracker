# coding: utf-8

from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.validation import is_2d_int_coords_valid
from supervisely.geometry.sliding_windows import SlidingWindows
from supervisely.collection.str_enum import StrEnum


class SlidingWindowBorderStrategy(StrEnum):
    """ """

    ADD_PADDING = "add_padding"
    """"""
    SHIFT_WINDOW = "shift_window"
    """"""
    CHANGE_SIZE = "change_size"
    """"""


class SlidingWindowsFuzzy(SlidingWindows):
    """ """

    def __init__(
        self, window_shape, min_overlap, strategy=str(SlidingWindowBorderStrategy.SHIFT_WINDOW)
    ):
        super().__init__(window_shape, min_overlap)
        if not SlidingWindowBorderStrategy.has_value(strategy):
            raise ValueError(
                "Unknown strategy {!r}. Allowed strategies: {}".format(
                    strategy, SlidingWindowBorderStrategy.values()
                )
            )
        self.strategy = strategy

    def get(self, source_shape):
        """ """
        if self.strategy == str(SlidingWindowBorderStrategy.SHIFT_WINDOW):
            yield from self.get_shift_window(source_shape)
        elif self.strategy == str(SlidingWindowBorderStrategy.ADD_PADDING):
            yield from self.get_add_padding(source_shape)
        elif self.strategy == str(SlidingWindowBorderStrategy.CHANGE_SIZE):
            yield from self.get_change_size(source_shape)
        else:
            raise NotImplementedError("Not implemented SW Strategy: {!r}".format(self.strategy))

    def get_shift_window(self, source_shape):
        """ """
        yield from super().get(source_shape)

    def get_add_padding(self, source_shape):
        """ """
        h = source_shape[0]
        w = source_shape[1]
        source_rect = Rectangle.from_size(source_shape)
        window_rect = Rectangle.from_size(self.window_shape)
        if not source_rect.contains(window_rect):
            raise RuntimeError("Sliding window: window is larger than source (image).")

        # hw_limit = tuple(source_shape[i] - self.window_shape[i] for i in (0, 1))
        # for wind_top in range(0, hw_limit[0] + self.stride[0], self.stride[0]):
            # for wind_left in range(0, hw_limit[1] + self.stride[1], self.stride[1]):
        for wind_top in range(0, h, self.stride[0]):
            for wind_left in range(0, w, self.stride[1]):
                roi = window_rect.translate(drow=wind_top, dcol=wind_left)
                yield roi

    def get_change_size(self, source_shape):
        """ """
        h = source_shape[0]
        w = source_shape[1]
        source_rect = Rectangle.from_size(source_shape)
        window_rect = Rectangle.from_size(self.window_shape)
        if not source_rect.contains(window_rect):
            raise RuntimeError("Sliding window: window is larger than source (image).")

        # hw_limit = tuple(source_shape[i] - self.window_shape[i] for i in (0, 1))
        # for wind_top in range(0, hw_limit[0] + self.stride[0], self.stride[0]):
        #     for wind_left in range(0, hw_limit[1] + self.stride[1], self.stride[1]):
        for wind_top in range(0, h, self.stride[0]):
            for wind_left in range(0, w, self.stride[1]):
                wind_bottom = min(wind_top + self.stride[0], source_shape[0])
                wind_right = min(wind_left + self.stride[1], source_shape[1])
                roi = Rectangle(wind_top, wind_left, wind_bottom - 1, wind_right - 1)
                if not source_rect.contains(roi):
                    raise RuntimeError("Sliding window: result crop bounds are invalid.")
                yield roi
