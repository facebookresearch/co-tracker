from supervisely.app.widgets import Container, OneOf, Text, Flexbox
import supervisely.nn.inference.instance_segmentation.dashboard.preview as preview


oneof_block = OneOf(preview.image_source)

preview_image_layout = Container(
    [preview.image_source_field, oneof_block],
)
