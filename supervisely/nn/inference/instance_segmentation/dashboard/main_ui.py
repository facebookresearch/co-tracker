from supervisely.app.widgets import Text, Select, Menu, Field, Container, Button
from supervisely.imaging.color import hex2rgb
from supervisely.nn.inference.instance_segmentation.dashboard.deploy_ui import (
    deploy_layout,
)
from supervisely.nn.inference.instance_segmentation.dashboard.classes_ui import (
    classes_layout,
)
from supervisely.nn.inference.instance_segmentation.dashboard.preview_image import (
    preview_image_layout,
)

l = Text(text="left part", status="success")


ttt = Text(text="some text", status="warning")
# sidebar = sly.app.widgets.Sidebar(left_pane=l, right_pane=item)

g1 = Menu.Group(
    "Model",
    [
        Menu.Item(
            title="Deployment / Run", content=deploy_layout, icon="zmdi zmdi-dns"
        ),
        Menu.Item(title="Classes", content=classes_layout, icon="zmdi zmdi-shape"),
        Menu.Item(title="Monitoring", content=l, icon="zmdi zmdi-chart"),
    ],
)
g2 = Menu.Group(
    "Preview predictions",
    [
        Menu.Item(title="Image", content=preview_image_layout, icon="zmdi zmdi-image"),
        Menu.Item(title="Video", content=ttt, icon="zmdi zmdi-youtube-play"),
    ],
)
g3 = Menu.Group(
    "Inference",
    [
        Menu.Item(
            title="Apply to images project",
            content=ttt,
            icon="zmdi zmdi-collection-folder-image",
        ),
        Menu.Item(
            title="Apply to videos project",
            content=ttt,
            icon="zmdi zmdi-collection-video",
        ),
    ],
)
menu = Menu(groups=[g1, g2, g3], index="Image")
# menu = sly.app.widgets.Menu(items=g1_items)
