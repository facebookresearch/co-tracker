from __future__ import annotations
from typing import List
from supervisely.app.widgets import Widget
from supervisely.api.project_api import ProjectInfo
from supervisely.project.project import Project
import supervisely.imaging.color as sly_color

"""
items = [
    Select.Item(label="CPU", value="cpu"),
    Select.Item(label="GPU 0", value="cuda:0"),
    Select.Item(value="option3"),
]
r = Select(items=items, filterable=True, placeholder="select me")

f0 = Field(r, "title0")
f1 = Field(r, "title1", "description1")
f2 = Field(r, "title2", "description2", title_url="/a/b")
f3 = Field(r, "title3", "description3", description_url="/a/b")
f4 = Field(r, "title4", "description4", title_url="/a/b", description_url="/a/b")
f5 = Field(r, "title5", "with icon", icon=Field.Icon(zmdi_class="zmdi zmdi-bike"))
f6 = Field(
    r,
    "title6",
    "with image",
    icon=Field.Icon(image_url="https://i.imgur.com/0E8d8bB.png"),
)

fields = Container([f0, f1, f2, f3, f4, f5, f6])

"""


class Field(Widget):
    class Icon:
        def __init__(
            self,
            zmdi_class=None,
            color_rgb: List[int, int, int] = None,
            bg_color_rgb: List[int, int, int] = None,
            image_url=None,
        ) -> Field.Icon:
            if zmdi_class is None and image_url is None:
                raise ValueError(
                    "One of the arguments has to be defined: zmdi_class or image_url"
                )
            if zmdi_class is not None and image_url is not None:
                raise ValueError(
                    "Only one of the arguments has to be defined: zmdi_class or image_url"
                )
            if image_url is not None and (
                color_rgb is not None or bg_color_rgb is not None
            ):
                raise ValueError(
                    "Arguments color_rgb / bg_color_rgb can not be used with image_url at the same time"
                )

            if image_url is None and color_rgb is None:
                color_rgb = [255, 255, 255]

            if image_url is None and bg_color_rgb is None:
                bg_color_rgb = [0, 154, 255]

            self._zmdi_class = zmdi_class
            self._color = color_rgb
            self._bg_color = bg_color_rgb
            self._image_url = image_url
            if self._color is not None:
                sly_color._validate_color(self._color)
            if self._bg_color is not None:
                sly_color._validate_color(self._bg_color)

        def to_json(self):
            res = {}
            if self._zmdi_class is not None:
                res["className"] = self._zmdi_class
                res["color"] = sly_color.rgb2hex(self._color)
                res["bgColor"] = sly_color.rgb2hex(self._bg_color)
            if self._image_url is not None:
                res["imageUrl"] = self._image_url
            return res

    def __init__(
        self,
        content: Widget,
        title: str,
        description: str = None,
        title_url: str = None,
        description_url: str = None,
        icon: Field.Icon = None,
        widget_id: str = None,
    ):
        self._title = title
        self._description = description
        self._title_url = title_url
        self._description_url = description_url
        self._icon = icon
        self._content = content
        if self._title_url is not None and self._title is None:
            raise ValueError(
                "Title can not be specified only as url without text value"
            )
        if self._description_url is not None and self._description is None:
            raise ValueError(
                "Description can not be specified only as url without text value"
            )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        res = {
            "title": self._title,
            "description": self._description,
            "title_url": self._title_url,
            "description_url": self._description_url,
            "icon": None,
        }
        if self._icon is not None:
            res["icon"] = self._icon.to_json()
        return res

    def get_json_state(self):
        return None
