import copy
from typing import List
import cv2
import imutils
import numpy as np
import supervisely as sly
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets


class ImageRegionSelector(Widget):
    class Routes:
        BBOX_CHANGED = "bbox-changed"
        POSITIVE_CHANGED = "positive-updated"
        NEGATIVE_CHANGED = "negative-updated"

    def __init__(
        self,
        image_info: sly.ImageInfo = None,
        mask: sly.Bitmap = None,
        mask_opacity: int = 50,
        bbox: List[int] = None,
        points_disabled: bool = False,
        widget_id: str = None,
        disabled: bool = False,
        widget_width: str = "100%",
        widget_height: str = "100%",
    ):
        self._image_info = None
        self._image_link = None
        self._image_name = None
        self._image_url = None
        self._image_hash = None
        self._image_size = None
        self._image_id = None
        self._image_width = None
        self._image_height = None
        self._dataset_id = None
        self._mask = None
        self._mask_opacity = mask_opacity
        self._original_bbox = None
        self._scaled_bbox = None
        self._disabled = disabled
        self._points_disabled = points_disabled
        self._widget_width = widget_width
        self._widget_height = widget_height
        self._bbox_changes_handled = False
        self._pos_points_changes_handled = False
        self._neg_points_changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)
        if image_info is not None:
            self.image_update(image_info)
        
        if mask is not None:
            self.set_mask(mask)
        
        if bbox is not None:
            self.set_bbox(bbox)

        script_path = "./sly/css/app/widgets/image_region_selector/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "1"] = "https://cdn.jsdelivr.net/npm/svg.js@2.7.1/dist/svg.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "2"] = "https://cdn.jsdelivr.net/npm/svg.select.js@3.0.1/dist/svg.select.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "3"] = "https://cdn.jsdelivr.net/npm/svg.resize.js@1.4.3/dist/svg.resize.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "4"] = "https://cdn.jsdelivr.net/npm/svg.draggable.js@2.2.2/dist/svg.draggable.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "5"] = "https://cdn.jsdelivr.net/npm/svg.panzoom.js@1.2.3/dist/svg.panzoom.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "6"] = "https://rawgit.com/nodeca/pako/1.0.11/dist/pako.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "7"] = "https://cdnjs.cloudflare.com/ajax/libs/uuid/8.3.2/uuidv4.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def image_update(self, image_info: sly.ImageInfo):
        self._image_info = image_info
        self._image_link = image_info.preview_url
        self._image_name = image_info.name
        self._image_url = image_info.preview_url
        self._image_hash = image_info.hash
        self._image_size = image_info.size
        self._image_id = image_info.id
        self._image_width = image_info.width
        self._image_height = image_info.height
        self._dataset_id = image_info.dataset_id

        padding_pix = int(max([image_info.width, image_info.height]) * 0.1)
        self._original_bbox = [
            [padding_pix, padding_pix],
            [image_info.width - padding_pix, image_info.height - padding_pix],
        ]
        self._scaled_bbox = self._original_bbox

        StateJson()[self.widget_id].update(self.get_json_state())
        StateJson().send_changes()
    
    def set_image(self, image_info: sly.ImageInfo):
        self._image_info = image_info
        self._image_link = image_info.preview_url
        self._image_name = image_info.name
        self._image_url = image_info.preview_url
        self._image_hash = image_info.hash
        self._image_size = image_info.size
        self._image_id = image_info.id
        self._image_width = image_info.width
        self._image_height = image_info.height
        self._dataset_id = image_info.dataset_id
        
        padding_pix = int(max([image_info.width, image_info.height]) * 0.1)
        self._original_bbox = [
            [padding_pix, padding_pix],
            [image_info.width - padding_pix, image_info.height - padding_pix],
        ]
        self._scaled_bbox = self._original_bbox
        
        StateJson()[self.widget_id].update(self.get_json_state())
        StateJson().send_changes()

    def get_image_info(self):
        return self._image_info

    def set_bbox(self, bbox):
        if bbox is None:
            padding_pix = int(max([self._image_width, self._image_height]) * 0.1)
            bbox = [
                [padding_pix, padding_pix],
                [self._image_width - padding_pix, self._image_height - padding_pix],
            ]

        self._original_bbox = bbox
        self._scaled_bbox = self._original_bbox

        StateJson()[self.widget_id].update(self.get_json_state())
        StateJson().send_changes()

    def get_bbox(self):
        return self._original_bbox

    def set_mask(self, bitmap: sly.Bitmap, color="#77e377"):
        mask = None
        if bitmap is not None:
            mask_base64 = bitmap.data_2_base64(bitmap.data)
            origin = [bitmap.origin.col, bitmap.origin.row]
            contours = copy.deepcopy(self._get_contours(mask_base64, origin))
            mask = {
                'data': mask_base64,
                'origin': origin,
                'color': color,
                'contour': contours
            }
        self._mask = mask
        self.update_state()
        StateJson().send_changes()

    def get_mask(self):
        return self._mask

    def bbox_changed(self, func, page_path=""):
        route_path = page_path + self.get_route_path(ImageRegionSelector.Routes.BBOX_CHANGED)
        server = self._sly_app.get_server()
        self._bbox_changes_handled = True

        @server.post(route_path)
        def _click():
            self.bbox_update()
            res = self._scaled_bbox
            func(res)

        return _click
    
    def positive_points_changed(self, func, page_path=""):
        route_path = page_path + self.get_route_path(ImageRegionSelector.Routes.POSITIVE_CHANGED)
        server = self._sly_app.get_server()
        self._pos_points_changes_handled = True

        @server.post(route_path)
        def _click():
            points = self.get_positive_points()
            func(points)

        return _click
    
    def negative_points_changed(self, func, page_path=""):
        route_path = page_path + self.get_route_path(ImageRegionSelector.Routes.NEGATIVE_CHANGED)
        server = self._sly_app.get_server()
        self._neg_points_changes_handled = True

        @server.post(route_path)
        def _click():
            points = self.get_negative_points()
            func(points)

        return _click

    def bbox_update(self):
        self._scaled_bbox = StateJson()[self.widget_id]["scaledBbox"]
        bboxes_padding = 0

        scaled_width, scaled_height = self.get_bbox_size(self._scaled_bbox)
        original_width, original_height = int(scaled_width / (1 + bboxes_padding)), int(
            scaled_height / (1 + bboxes_padding)
        )

        div_width, div_height = (scaled_width - original_width) // 2, (
            scaled_height - original_height
        ) // 2

        self._original_bbox[0][0] = self._scaled_bbox[0][0] + div_width
        self._original_bbox[0][1] = self._scaled_bbox[0][1] + div_height
        self._original_bbox[1][0] = self._scaled_bbox[1][0] - div_width
        self._original_bbox[1][1] = self._scaled_bbox[1][1] - div_height
        StateJson().send_changes()

    def get_json_data(self):
        return {
            "disabled": self._disabled,
        }

    def get_json_state(self):
        return {
            "imageLink": self._image_link,
            "imageName": self._image_name,
            "imageUrl": self._image_url,
            "imageHash": self._image_hash,
            "imageSize": self._image_size,
            "imageId": self._image_id,
            "imageWidth": self._image_width,
            "imageHeight": self._image_height,
            "mask": self._mask,
            "mask_opacity": self._mask_opacity,
            "datasetId": self._dataset_id,
            "originalBbox": self._original_bbox,
            "scaledBbox": self._scaled_bbox,
            "disabled": self._disabled,
            "pointsDisabled": self._points_disabled,
            "widget_width": self._widget_width,
            "widget_height": self._widget_height,
            "widget_id": self.widget_id,
        }

    @property
    def is_empty(self):
        return not self._image_info is None

    def get_bbox_size(self, current_bbox):
        box_width = current_bbox[1][0] - current_bbox[0][0]
        box_height = current_bbox[1][1] - current_bbox[0][1]
        return box_width, box_height

    def add_bbox_padding(self, padding_coefficient=0):
        padding_coefficient /= 100

        original_w, original_h = self.get_bbox_size(current_bbox=self._original_bbox)
        additional_w, additional_h = (
            int(original_w * padding_coefficient // 2),
            int(original_h * padding_coefficient // 2),
        )

        self._scaled_bbox[0][0] = (
            self._original_bbox[0][0] - additional_w
            if self._original_bbox[0][0] - additional_w > 0
            else 0
        )
        self._scaled_bbox[0][1] = (
            self._original_bbox[0][1] - additional_h
            if self._original_bbox[0][1] - additional_h > 0
            else 0
        )
        self._scaled_bbox[1][0] = (
            self._original_bbox[1][0] + additional_w
            if self._original_bbox[1][0] + additional_w < self._image_size[0]
            else self._image_size[0] - 1
        )
        self._scaled_bbox[1][1] = (
            self._original_bbox[1][1] + additional_h
            if self._original_bbox[1][1] + additional_h < self._image_size[1]
            else self._image_size[1] - 1
        )
        StateJson().send_changes()

    def get_relative_coordinates(self, abs_coordinates):
        box_width, box_height = self.get_bbox_size(current_bbox=self._scaled_bbox)
        return {
            "x": (abs_coordinates["position"][0][0] - self._scaled_bbox[0][0]) / box_width,
            "y": (abs_coordinates["position"][0][1] - self._scaled_bbox[0][1]) / box_height,
        }

    def _get_contours(self, base64mask, origin_shift):
        test_mask = np.asarray(sly.Bitmap.base64_2_data(base64mask)).astype(np.uint8) * 255
        thresholded_mask = cv2.threshold(test_mask, 100, 255, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(thresholded_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        reshaped_contours = []
        for contour in contours:
            reshaped_contour = contour.reshape(contour.shape[0], 2)

            reshaped_contour[:, 0] += origin_shift[0]
            reshaped_contour[:, 1] += origin_shift[1]

            reshaped_contours.append(reshaped_contour.tolist())

        return reshaped_contours
    
    def get_positive_points(self):
        return StateJson()[self.widget_id]["positivePoints"]

    def get_negative_points(self):
        return StateJson()[self.widget_id]["negativePoints"]
