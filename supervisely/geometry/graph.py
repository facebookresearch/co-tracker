# coding: utf-8

# docs
from __future__ import annotations
import numpy as np
import cv2
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union
from supervisely.geometry.image_rotator import ImageRotator


from supervisely.imaging.color import rgb2hex, hex2rgb, _validate_color
from supervisely.io.json import JsonSerializable

from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import LABELER_LOGIN, CREATED_AT, UPDATED_AT, ID, CLASS_ID


EDGES = "edges"
NODES = "nodes"

DISABLED = "disabled"
LOC = "loc"

DST = "dst"
SRC = "src"
COLOR = "color"


class Node(JsonSerializable):
    """
    Node for a single :class:`GraphNodes<GraphNodes>`.

    :param location: PointLocation object.
    :type location: PointLocation
    :param disabled: Determines whether to display the Node when drawing or not.
    :type disabled: bool, optional
    :param label: str
    :param row: int
    :param col: int
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.geometry.graph import Node

        vertex = Node(sly.PointLocation(5, 5))
    """

    def __init__(
        self,
        location: Optional[PointLocation] = None,
        disabled: Optional[bool] = False,
        label: Optional[str] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ):
        if None not in (location, row, col) or all(item is None for item in (location, row, col)):
            raise ValueError("Either location or row and col must be specified")
        self._location = location
        self._disabled = disabled
        self._label = label
        if None not in [row, col]:
            self._location = PointLocation(row, col)

    @property
    def location(self) -> PointLocation:
        """
        Location of Node.

        :return: PointLocation object
        :rtype: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`
        """
        return self._location

    @property
    def disabled(self) -> bool:
        """
        Display the Node when drawing or not.

        :return: Boolean
        :rtype: :class:`bool`
        """
        return self._disabled

    @classmethod
    def from_json(cls, data: Dict) -> Node:
        """
        Convert a json dict to Node. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Node in json format as a dict.
        :type data: dict
        :return: Node object
        :rtype: :class:`Node<Node>`
        :Usage example:

         .. code-block:: python

            vertex_json = {
                "loc": [5, 5]
            }
            vertex = Node.from_json(vertex_json)
        """
        # TODO validations
        loc = data[LOC]
        return cls(
            location=PointLocation(row=loc[1], col=loc[0]), disabled=data.get(DISABLED, False)
        )

    def to_json(self) -> Dict:
        """
        Convert the Node to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.graph import Node

            vertex = Node(sly.PointLocation(5, 5))
            vertex_json = vertex.to_json()
            print(vertex_json)
            # Output: {
            #    "loc": [5, 5]
            # }
        """
        result = {LOC: [self._location.col, self._location.row]}
        if self.disabled:
            result[DISABLED] = True
        return result

    def transform_location(self, transform_fn):
        """
        :param transform_fn: function to convert location
        :return: Node class object with the changed location attribute using the given function
        """
        return Node(transform_fn(self._location), disabled=self.disabled)


def _maybe_transform_colors(elements, process_fn):
    """
    Function _maybe_transform_colors convert some list of parameters using the given function
    :param elements: list of elements
    :param process_fn: function to convert
    """
    for elem in elements:
        if COLOR in elem:
            elem[COLOR] = process_fn(elem[COLOR])


class GraphNodes(Geometry):
    """
    GraphNodes geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`GraphNodes<GraphNodes>` class object is immutable.

    :param nodes: Dict or List containing nodes of graph
    :type nodes: dict
    :param sly_id: GraphNodes ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which GraphNodes belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created GraphNodes.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when GraphNodes was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when GraphNodes was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.geometry.graph import Node, GraphNodes

        vertex_1 = Node(sly.PointLocation(5, 5))
        vertex_2 = Node(sly.PointLocation(100, 100))
        vertex_3 = Node(sly.PointLocation(200, 250))
        nodes = {0: vertex_1, 1: vertex_2, 2: vertex_3}
        figure = GraphNodes(nodes)
    """

    @staticmethod
    def geometry_name():
        return "graph"

    def __init__(
        self,
        nodes: Union[Dict[str, Dict], List],
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[int] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):

        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        self._nodes = nodes
        if isinstance(nodes, list):
            self._nodes = {}
            for i, node in enumerate(nodes):
                if node._label is not None:
                    self._nodes[node._label] = Node(node._location, node._disabled)
                else:
                    self._nodes[str(i)] = Node(node._location, node._disabled)

    @property
    def nodes(self) -> Dict[str, Dict]:
        """
        Copy of GraphNodes nodes.

        :return: GraphNodes nodes
        :rtype: :class:`dict`
        """
        return self._nodes.copy()

    @classmethod
    def from_json(cls, data: Dict[str, Dict]) -> GraphNodes:
        """
        Convert a json dict to GraphNodes. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: GraphNodes in json format as a dict.
        :type data: dict
        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`
        :Usage example:

         .. code-block:: python

            figure_json = {
                "nodes": {
                    "0": {
                        "loc": [5, 5]
                    },
                    "1": {
                        "loc": [100, 100]
                    },
                    "2": {
                        "loc": [250, 200]
                    }
                }
            }
            from supervisely.geometry.graph import GraphNodes
            figure = GraphNodes.from_json(figure_json)
        """
        nodes = {node_id: Node.from_json(node_json) for node_id, node_json in data["nodes"].items()}
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return GraphNodes(
            nodes=nodes,
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def to_json(self) -> Dict[str, Dict]:
        """
        Convert the GraphNodes to list. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.graph import Node, GraphNodes

            vertex_1 = Node(sly.PointLocation(5, 5))
            vertex_2 = Node(sly.PointLocation(100, 100))
            vertex_3 = Node(sly.PointLocation(200, 250))
            nodes = {0: vertex_1, 1: vertex_2, 2: vertex_3}
            figure = GraphNodes(nodes)

            figure_json = figure.to_json()
            print(figure_json)
            # Output: {
            #    "nodes": {
            #        "0": {
            #            "loc": [5, 5]
            #        },
            #        "1": {
            #            "loc": [100, 100]
            #        },
            #        "2": {
            #            "loc": [250, 200]
            #        }
            #    }
            # }
        """
        res = {NODES: {node_id: node.to_json() for node_id, node in self._nodes.items()}}
        self._add_creation_info(res)
        return res

    def crop(self, rect: Rectangle) -> List[GraphNodes]:
        """
        Crops current GraphNodes.

        :param rect: Rectangle object for crop.
        :type rect: Rectangle
        :return: List of GraphNodes objects
        :rtype: :class:`List[GraphNodes]`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            crop_figures = figure.crop(sly.Rectangle(0, 0, 300, 350))
        """
        is_all_nodes_inside = all(
            rect.contains_point_location(node.location) for node in self._nodes.values()
        )
        return [self] if is_all_nodes_inside else []

    def relative_crop(self, rect: Rectangle) -> List[GraphNodes]:
        """
        Crops current GraphNodes with given rectangle and shifts it on value of rectangle left top angle.

        :param rect: Rectangle object for crop.
        :type rect: Rectangle
        :return: List of GraphNodes objects
        :rtype: :class:`List[GraphNodes]<GraphNodes>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            rel_crop_figures = figure.relative_crop(sly.Rectangle(0, 0, 300, 350))
        """
        return [geom.translate(drow=-rect.top, dcol=-rect.left) for geom in self.crop(rect)]

    def transform(self, transform_fn) -> GraphNodes:
        """
        :param transform_fn: Function to convert GraphNodes.
        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`
        """
        return GraphNodes(
            nodes={node_id: transform_fn(node) for node_id, node in self._nodes.items()}
        )

    def transform_locations(self, transform_fn) -> GraphNodes:
        """
        :param transform_fn: Function to convert GraphNodes location.
        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`
        """
        return self.transform(lambda kp: kp.transform_location(transform_fn))

    def resize(self, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> GraphNodes:
        """
        Resizes current GraphNodes.

        :param in_size: Input image size (height, width) to which belongs GraphNodes.
        :type in_size: Tuple[int, int]
        :param out_size: Desired output image size (height, width) to which belongs GraphNodes.
        :type out_size: Tuple[int, int]
        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`

        :Usage Example:

         .. code-block:: python

            # Remember that GraphNodes class object is immutable, and we need to assign new instance of Rectangle to a new variable
            in_height, in_width = 300, 400
            out_height, out_width = 600, 800
            resize_figure = figure.resize((in_height, in_width), (out_height, out_width))
        """
        return self.transform_locations(lambda p: p.resize(in_size, out_size))

    def scale(self, factor: float) -> GraphNodes:
        """
        Scales current GraphNodes.

        :param factor: Scale parameter.
        :type factor: float
        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`

        :Usage Example:

         .. code-block:: python

            # Remember that GraphNodes class object is immutable, and we need to assign new instance of Rectangle to a new variable
            scale_figure = figure.scale(0.75)
        """
        return self.transform_locations(lambda p: p.scale(factor))

    def translate(self, drow: int, dcol: int) -> GraphNodes:
        """
        Translates current GraphNodes.

        :param drow: Horizontal shift.
        :type drow: int
        :param dcol: Vertical shift.
        :type dcol: int
        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`

        :Usage Example:

         .. code-block:: python

            # Remember that GraphNodes class object is immutable, and we need to assign new instance of Rectangle to a new variable
            translate_figure = figure.translate(150, 250)
        """
        return self.transform_locations(lambda p: p.translate(drow, dcol))

    def rotate(self, rotator: ImageRotator) -> GraphNodes:
        """
        Rotates current GraphNodes.

        :param rotator: ImageRotator object for rotation.
        :type rotator: ImageRotator
        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`

        :Usage Example:

         .. code-block:: python

            from supervisely.geometry.image_rotator import ImageRotator

            # Remember that GraphNodes class object is immutable, and we need to assign new instance of Rectangle to a new variable
            height, width = 300, 400
            rotator = ImageRotator((height, width), 25)
            rotate_figure = figure.rotate(rotator)
        """
        return self.transform_locations(lambda p: p.rotate(rotator))

    def fliplr(self, img_size: Tuple[int, int]) -> GraphNodes:
        """
        Flips current GraphNodes in horizontal.

        :param img_size: Input image size (height, width) to which belongs GraphNodes.
        :type img_size: Tuple[int, int]
        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`

        :Usage Example:

         .. code-block:: python

            # Remember that GraphNodes class object is immutable, and we need to assign new instance of Rectangle to a new variable
            height, width = 300, 400
            fliplr_figure = figure.fliplr((height, width))
        """
        return self.transform_locations(lambda p: p.fliplr(img_size))

    def flipud(self, img_size: Tuple[int, int]) -> GraphNodes:
        """
        Flips current GraphNodes in vertical.

        :param img_size: Input image size (height, width) to which belongs GraphNodes.
        :type img_size: Tuple[int, int]
        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`

        :Usage Example:

         .. code-block:: python

            # Remember that GraphNodes class object is immutable, and we need to assign new instance of Rectangle to a new variable
            height, width = 300, 400
            flipud_figure = figure.flipud((height, width))
        """
        return self.transform_locations(lambda p: p.flipud(img_size))

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """
        Draws the graph contour on a given bitmap canvas
        :param bitmap: numpy array
        :param color: tuple or list of integers
        :param thickness: int
        :param config: drawing config specific to a concrete subclass, e.g. per edge colors
        """
        self.draw_contour(bitmap, color, thickness, config=config)

    @staticmethod
    def _get_nested_or_default(dict, keys_path, default=None):
        """
        _get_nested_or_default
        """
        result = dict
        for key in keys_path:
            if result is not None:
                result = result.get(key, None)
        return result if result is not None else default

    def _draw_contour_impl(self, bitmap, color=None, thickness=1, config=None):
        """
        _draw_contour_impl
        """
        if config is not None:
            # If a config with edges and colors is passed, make sure it is
            # consistent with the our set of points.
            self.validate(self.geometry_name(), config)

        # Draw edges first so that nodeas are then drawn on top.
        for edge in self._get_nested_or_default(config, [EDGES], []):
            src = self._nodes.get(edge[SRC], None)
            dst = self._nodes.get(edge[DST], None)
            if (
                (src is not None)
                and (not src.disabled)
                and (dst is not None)
                and (not dst.disabled)
            ):
                edge_color = edge.get(COLOR, color)
                cv2.line(
                    bitmap,
                    (src.location.col, src.location.row),
                    (dst.location.col, dst.location.row),
                    tuple(edge_color),
                    thickness,
                )

        nodes_config = self._get_nested_or_default(config, [NODES])
        for node_id, node in self._nodes.items():
            if not node.disabled:
                effective_color = self._get_nested_or_default(nodes_config, [node_id, COLOR], color)
                Point.from_point_location(node.location).draw(
                    bitmap=bitmap, color=effective_color, thickness=thickness, config=None
                )

    @property
    def area(self) -> float:
        """
        GraphNodes area.

        :return: Area of current GraphNodes, always 0.0
        :rtype: :class:`float`

        :Usage Example:

         .. code-block:: python

            print(figure.area)
            # Output: 0.0
        """
        return 0.0

    def to_bbox(self) -> Rectangle:
        """
        Create Rectangle object from current GraphNodes.

        :return: Rectangle object
        :rtype: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`

        :Usage Example:

         .. code-block:: python

            rectangle = figure.to_bbox()
        """
        return Rectangle.from_geometries_list(
            [Point.from_point_location(node.location) for node in self._nodes.values()]
        )

    def clone(self) -> GraphNodes:
        """
        Makes a copy of the GraphNodes.

        :return: GraphNodes object
        :rtype: :class:`GraphNodes<GraphNodes>`

        :Usage Example:

         .. code-block:: python

            # Remember that GraphNodes class object is immutable, and we need to assign new instance of PointLocation to a new variable
            new_figure = figure.clone()
        """
        return self

    def validate(self, name: str, settings: Dict) -> None:
        """
        Checks the graph for correctness and compliance with the template
        """
        super().validate(name, settings)
        # TODO template self-consistency checks.

        nodes_not_in_template = set(self._nodes.keys()) - set(settings[NODES].keys())
        if len(nodes_not_in_template) > 0:
            raise ValueError(
                "Graph contains nodes not declared in the template: {!r}.".format(
                    nodes_not_in_template
                )
            )

    @staticmethod
    def _transform_config_colors(config, transform_fn):
        """
        Transform colors of edges and nodes in graph template
        :param config: dictionary(graph template)
        :param transform_fn: function to convert
        :return: dictionary(graph template)
        """
        if config is None:
            return None

        result = deepcopy(config)
        _maybe_transform_colors(result.get(EDGES, []), transform_fn)
        _maybe_transform_colors(result[NODES].values(), transform_fn)
        return result

    @staticmethod
    def config_from_json(config: Dict) -> Dict:
        """
        Convert graph template from json format
        :param config: dictionary(graph template) in json format
        :return: dictionary(graph template)
        """
        return GraphNodes._transform_config_colors(config, hex2rgb)

    @staticmethod
    def config_to_json(config: Dict) -> Dict:
        """
        Convert graph template in json format
        :param config: dictionary(graph template)
        :return: dictionary(graph template) in json format
        """
        return GraphNodes._transform_config_colors(config, rgb2hex)

    @classmethod
    def allowed_transforms(cls):
        """
        allowed_transforms
        """
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.rectangle import Rectangle

        return [AnyGeometry, Rectangle]


class KeypointsTemplate(GraphNodes, Geometry):
    def __init__(self):
        self._config = {"nodes": {}, "edges": []}
        self._point_names = []

    def add_point(self, label: str, row: int, col: int, color: list = [0, 0, 255]):
        _validate_color(color)
        if label in self._config["nodes"]:
            raise KeyError(f"Label {label} already exists in the graph")
        self._point_names.append(label)
        self._config["nodes"][label] = {
            "label": label,
            "loc": [row, col],
            "color": color,
        }

    def add_edge(self, src: str, dst: str, color: list = [0, 255, 0]):
        _validate_color(color)
        for elem in (src, dst):
            if elem not in self._config["nodes"]:
                raise ValueError(f"There is no such node in the graph: {elem}")
        self._config["edges"].append({"src": src, "dst": dst, "color": color})

    def get_nodes(self):
        self._nodes = {}
        for node in self._config["nodes"]:
            loc = self._config["nodes"][node]["loc"]
            self._nodes[node] = Node(PointLocation(loc[1], loc[0]))

    def draw(self, image: np.ndarray, thickness=7):
        self.get_nodes()
        self._draw_bool_compatible(
            self._draw_impl,
            bitmap=image,
            color=[0, 255, 0],
            thickness=thickness,
            config=self._config,
        )

    def to_json(self):
        return self.config_to_json(self._config)

    @property
    def config(self):
        return self._config

    @property
    def point_names(self):
        """
        Return point names in order in which they were added
        """
        return self._point_names
