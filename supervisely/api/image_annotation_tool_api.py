# coding: utf-8

import os
import time
from collections import defaultdict, OrderedDict
import json

from supervisely.api.module_api import ApiField, ModuleApiBase, ModuleWithStatus, WaitingTimeExceeded
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from supervisely.io.fs import get_file_name, ensure_base_path, get_file_hash
from supervisely.collection.str_enum import StrEnum
from supervisely._utils import batched


class ImageAnnotationToolAction(StrEnum):
    SET_FIGURE = 'figures/setFigure'
    """"""
    NEXT_IMAGE = 'images/nextImage'
    """"""
    PREV_IMAGE = 'images/prevImage'
    """"""
    SET_IMAGE = 'images/setImage'
    """"""
    ZOOM_TO_FIGURE = 'scene/zoomToObject'
    """"""


class ImageAnnotationToolApi(ModuleApiBase):
    def set_figure(self, session_id, figure_id):
        """
        """
        return self._act(session_id, ImageAnnotationToolAction.SET_FIGURE, {ApiField.FIGURE_ID: figure_id})

    def next_image(self, session_id, image_id):
        """
        """
        return self._act(session_id, ImageAnnotationToolAction.NEXT_IMAGE, {ApiField.IMAGE_ID: image_id})

    def prev_image(self, session_id, image_id):
        """
        """
        return self._act(session_id, ImageAnnotationToolAction.PREV_IMAGE, {ApiField.IMAGE_ID: image_id})

    def set_image(self, session_id, image_id):
        """
        """
        return self._act(session_id, ImageAnnotationToolAction.SET_IMAGE, {ApiField.IMAGE_ID: image_id})

    def zoom_to_figure(self, session_id, figure_id, zoom_factor=1):
        """
        """
        return self._act(session_id, ImageAnnotationToolAction.ZOOM_TO_FIGURE,
                         {ApiField.FIGURE_ID: figure_id, ApiField.ZOOM_FACTOR: zoom_factor})

    def _act(self, session_id: int, action: ImageAnnotationToolAction, payload: dict):
        """
        """
        data = {ApiField.SESSION_ID: session_id, ApiField.ACTION: str(action), ApiField.PAYLOAD: payload}
        resp = self._api.post('/annotation-tool.run-action', data)
        return resp.json()



    # {
    #     "sessionId": "940c4ec7-3818-420b-9277-ab3c820babe5",
    #     "action": "scene/setViewport",
    #     "payload": {
    #         "viewport": {
    #             "offsetX": -461, # width
    #             "offsetY": -1228, # height
    #             "zoom": 1.7424000000000024
    #         }
    #     }
    # }

    # {
    #     "sessionId": "940c4ec7-3818-420b-9277-ab3c820babe5",
    #     "action": "scene/zoomToObject",
    #     "payload": {
    #         "figureId": 22129,
    #         "zoomFactor": 1.5
    #     }
    # }