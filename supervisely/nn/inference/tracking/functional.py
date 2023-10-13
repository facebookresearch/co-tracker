import numpy as np
from typing import List, Tuple

import supervisely as sly
from supervisely.nn.prediction_dto import PredictionPoint


def numpy_to_dto_point(points: np.ndarray, class_name: str) -> List[PredictionPoint]:
    assert points.shape[-1] == 2
    return [PredictionPoint(class_name=class_name, col=p[1], row=p[0]) for p in points]


def graph_to_dto_points(graph: sly.GraphNodes) -> Tuple[List[PredictionPoint], List[str]]:
    nodes = graph.nodes
    node_ids = [nid for nid in nodes.keys()]  # [str]
    nodes: List[sly.Node] = [node for node in nodes.values()]
    point_locations = [node.location for node in nodes]  # [sly.Pointlocation]
    points = [PredictionPoint(class_name="graph", col=pl.col, row=pl.row) for pl in point_locations]
    return points, node_ids


def dto_points_to_point_location(points: List[PredictionPoint]) -> List[sly.PointLocation]:
    return [sly.PointLocation(row=p.row, col=p.col) for p in points]


def exteriors_to_sly_polygons(exteriors: List[List[sly.PointLocation]]) -> List[sly.Polygon]:
    return [sly.Polygon(exterior=exterior) for exterior in exteriors]


def exterior_to_sly_polyline(exteriors: List[List[sly.PointLocation]]) -> List[sly.Polyline]:
    return [sly.Polyline(exterior=exterior) for exterior in exteriors]


def nodes_to_sly_graph(nodes: List[List[sly.Node]]) -> List[sly.GraphNodes]:
    return [sly.GraphNodes(tn) for tn in nodes]


def dto_points_to_sly_points(points: List[PredictionPoint]) -> List[sly.Point]:
    return [sly.Point(row=p.row, col=p.col) for p in points]


def dto_points_to_sly_nodes(points: List[PredictionPoint], label: str) -> List[sly.Node]:
    nodes = []
    for point in points:
        nodes.append(sly.Node(label=label, col=point.col, row=point.row))
    return nodes
