from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from typing import Optional, Union


class Image(Widget):
    def __init__(
        self,
        url: str = "",
        height: Optional[Union[int, str]] = None,
        width: Optional[Union[int, str]] = None,
        widget_id: str = None,
    ):
        self._url = url
        self._height, self._width = self._check_image_size(height=height, width=width)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "url": self._url,
            "height": self._height,
            "width": self._width,
        }

    def get_json_state(self):
        return None

    def set(
        self,
        url: str,
        height: Optional[Union[int, str]] = None,
        width: Optional[Union[int, str]] = None,
    ):
        self._update(url=url, height=height, width=width)

    def clean_up(self):
        self._update(url="", height=None, width=None)

    def set_image_size(
        self,
        height: Optional[Union[int, str]] = None,
        width: Optional[Union[int, str]] = None,
    ):
        self._update(url=self._url, height=height, width=width)

    def _update(
        self,
        url: str = "",
        height: Optional[Union[int, str]] = None,
        width: Optional[Union[int, str]] = None,
    ):
        self._url = url
        self._height, self._width = self._check_image_size(height=height, width=width)
        DataJson()[self.widget_id]["url"] = self._url
        DataJson()[self.widget_id]["height"] = self._height
        DataJson()[self.widget_id]["width"] = self._width
        DataJson().send_changes()

    def _check_image_size(
        self,
        height: Optional[Union[int, str]],
        width: Optional[Union[int, str]],
    ):
        if height is None and width is None:
            return "auto", "100%"

        def _check_single_size(size: Optional[Union[int, str]]):
            if size is None:
                return "auto"
            elif isinstance(size, int):
                return f"{size}px"
            elif isinstance(size, str):
                if size.endswith("px") or size.endswith("%") or size == "auto":
                    return size
                else:
                    raise ValueError(f"size must be in pixels or percent, got '{size}'")
            else:
                raise ValueError(f"size must be int or str, got '{type(size)}'")

        height = _check_single_size(size=height)
        width = _check_single_size(size=width)
        return height, width
