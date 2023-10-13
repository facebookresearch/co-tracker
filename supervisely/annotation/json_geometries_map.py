# coding: utf-8
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.mask_3d import Mask3D
from supervisely.geometry.cuboid import Cuboid
from supervisely.geometry.point import Point
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.geometry.point_3d import Point3d
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh


_INPUT_GEOMETRIES = [
    Bitmap,
    Mask3D,
    Cuboid,
    Point,
    Polygon,
    Polyline,
    Rectangle,
    GraphNodes,
    AnyGeometry,
    Cuboid3d,
    Pointcloud,
    Point3d,
    MultichannelBitmap,
    ClosedSurfaceMesh,
]
_JSON_SHAPE_TO_GEOMETRY_TYPE = {
    geometry.geometry_name(): geometry for geometry in _INPUT_GEOMETRIES
}


def GET_GEOMETRY_FROM_STR(figure_shape: str):
    """
    The function create geometry class object from given string
    """
    geometry = _JSON_SHAPE_TO_GEOMETRY_TYPE[figure_shape]
    return geometry
