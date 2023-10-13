import numpy as np
from typing import Optional, List, Dict, Union


class Prediction:
    def __init__(self, class_name):
        self.class_name = class_name


class PredictionMask(Prediction):
    def __init__(self, class_name: str, mask: np.ndarray, score: Optional[float] = None):
        super(PredictionMask, self).__init__(class_name=class_name)
        self.mask = mask
        self.score = score


class PredictionBBox(Prediction):
    def __init__(self, class_name: str, bbox_tlbr: List[int], score: Optional[float]):
        super(PredictionBBox, self).__init__(class_name=class_name)
        self.bbox_tlbr = bbox_tlbr
        self.score = score


class PredictionSegmentation(Prediction):
    def __init__(self, mask: np.ndarray):
        self.mask = mask


class PredictionKeypoints(Prediction):
    def __init__(self, class_name: str, labels: List[str], coordinates: List[float]):
        super(PredictionKeypoints, self).__init__(class_name=class_name)
        self.labels = labels
        self.coordinates = coordinates


class PredictionPoint(Prediction):
    def __init__(self, class_name: str, col: int, row: int):
        super().__init__(class_name=class_name)
        self.col = col
        self.row = row
