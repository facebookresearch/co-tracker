from supervisely.app.widgets import (
    Card,
    Text,
    Select,
    Field,
    Container,
    ObjectClassView,
    Checkbox,
    ObjectClassesList,
    Widget,
)
from supervisely.annotation.obj_class import ObjClass


from supervisely.geometry.bitmap import Bitmap
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

# model_classes = Container(
#     [
#         Checkbox(ObjectClassView(ObjClass("person", AnyGeometry))),
#         Checkbox(ObjectClassView(ObjClass("person", Rectangle))),
#         Checkbox(ObjectClassView(ObjClass("person", Polygon))),
#         Checkbox(ObjectClassView(ObjClass("person", Bitmap))),
#         Checkbox(ObjectClassView(ObjClass("person", Point))),
#         Checkbox(ObjectClassView(ObjClass("person", Polyline))),
#         Checkbox(ObjectClassView(ObjClass("person", Cuboid))),
#         # ObjClassView(ObjClass("person", GraphNodes, geometry_config={"1": "2"})),
#         Checkbox(ObjectClassView(ObjClass("person", Point3d))),
#         Checkbox(ObjectClassView(ObjClass("person", Cuboid3d))),
#         Checkbox(ObjectClassView(ObjClass("person", Pointcloud))),
#         Checkbox(ObjectClassView(ObjClass("person", Cuboid3d))),
#         Checkbox(ObjectClassView(ObjClass("person", Pointcloud))),
#         Checkbox(ObjectClassView(ObjClass("person", ClosedSurfaceMesh))),
#         Checkbox(ObjectClassView(ObjClass("person", MultichannelBitmap))),
#     ],
#     direction="horizontal",
#     overflow="wrap",
# )


_classes = [
    ObjClass("person1", AnyGeometry),
    ObjClass("person2", Rectangle),
    ObjClass("person3", Polygon),
    ObjClass("person4", Bitmap),
    ObjClass("person5", Point),
    ObjClass("person6", Polyline),
    ObjClass("person7", Cuboid),
    ObjClass("person8", Point3d),
    ObjClass("person9", Cuboid3d),
    ObjClass("person10", Pointcloud),
    ObjClass("person11", Cuboid3d),
    ObjClass("person12", Pointcloud),
    ObjClass("person13", ClosedSurfaceMesh),
    ObjClass("person14", MultichannelBitmap),
]

model_classes = ObjectClassesList(object_classes=_classes, columns=3, selectable=True)

text = Text("123")
classes_card = Card(
    "Model classes",
    "Model predicts the following classes",
    content=model_classes,
)

classes_layout = Container(
    [
        classes_card,
    ],
    direction="vertical",
)
