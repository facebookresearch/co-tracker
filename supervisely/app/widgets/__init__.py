from supervisely.app.widgets.widget import ConditionalWidget, ConditionalItem, DynamicWidget
from supervisely.app.widgets.widget import Widget, generate_id
from supervisely.app.widgets.radio_table.radio_table import RadioTable
from supervisely.app.widgets.notification_box.notification_box import NotificationBox
from supervisely.app.widgets.done_label.done_label import DoneLabel
from supervisely.app.widgets.sly_tqdm.sly_tqdm import SlyTqdm, Progress
from supervisely.app.widgets.grid_gallery.grid_gallery import GridGallery
from supervisely.app.widgets.classes_table.classes_table import ClassesTable
from supervisely.app.widgets.classic_table.classic_table import ClassicTable
from supervisely.app.widgets.confusion_matrix.confusion_matrix import ConfusionMatrix
from supervisely.app.widgets.project_selector.project_selector import ProjectSelector
from supervisely.app.widgets.element_button.element_button import ElementButton
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.project_thumbnail.project_thumbnail import ProjectThumbnail
from supervisely.app.widgets.apexchart.apexchart import Apexchart
from supervisely.app.widgets.line_chart.line_chart import LineChart
from supervisely.app.widgets.scatter_chart.scatter_chart import ScatterChart
from supervisely.app.widgets.heatmap_chart.heatmap_chart import HeatmapChart
from supervisely.app.widgets.treemap_chart.treemap_chart import TreemapChart
from supervisely.app.widgets.table.table import Table
from supervisely.app.widgets.labeled_image.labeled_image import LabeledImage
from supervisely.app.widgets.text.text import Text
from supervisely.app.widgets.sidebar.sidebar import Sidebar
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.card.card import Card
from supervisely.app.widgets.select.select import Select, SelectString
from supervisely.app.widgets.menu.menu import Menu
from supervisely.app.widgets.field.field import Field
from supervisely.app.widgets.input_number.input_number import InputNumber
from supervisely.app.widgets.video.video import Video
from supervisely.app.widgets.object_class_view.object_class_view import ObjectClassView
from supervisely.app.widgets.checkbox.checkbox import Checkbox
from supervisely.app.widgets.grid.grid import Grid
from supervisely.app.widgets.object_classes_list.object_classes_list import (
    ObjectClassesList,
)
from supervisely.app.widgets.empty.empty import Empty
from supervisely.app.widgets.one_of.one_of import OneOf
from supervisely.app.widgets.flexbox.flexbox import Flexbox
from supervisely.app.widgets.input.input import Input
from supervisely.app.widgets.select_team.select_team import SelectTeam
from supervisely.app.widgets.select_workspace.select_workspace import SelectWorkspace
from supervisely.app.widgets.select_project.select_project import SelectProject
from supervisely.app.widgets.select_dataset.select_dataset import SelectDataset
from supervisely.app.widgets.select_item.select_item import SelectItem
from supervisely.app.widgets.select_app_session.select_app_session import SelectAppSession
from supervisely.app.widgets.identity.identity import Identity
from supervisely.app.widgets.dataset_thumbnail.dataset_thumbnail import DatasetThumbnail
from supervisely.app.widgets.select_tag_meta.select_tag_meta import SelectTagMeta
from supervisely.app.widgets.video_thumbnail.video_thumbnail import VideoThumbnail
from supervisely.app.widgets.tabs.tabs import Tabs
from supervisely.app.widgets.radio_tabs.radio_tabs import RadioTabs
from supervisely.app.widgets.train_val_splits.train_val_splits import TrainValSplits
from supervisely.app.widgets.editor.editor import Editor
from supervisely.app.widgets.textarea.textarea import TextArea
from supervisely.app.widgets.destination_project.destination_project import DestinationProject
from supervisely.app.widgets.image.image import Image
from supervisely.app.widgets.random_splits_table.random_splits_table import RandomSplitsTable
from supervisely.app.widgets.video_player.video_player import VideoPlayer
from supervisely.app.widgets.radio_group.radio_group import RadioGroup
from supervisely.app.widgets.switch.switch import Switch
from supervisely.app.widgets.input_tag.input_tag import InputTag
from supervisely.app.widgets.file_viewer.file_viewer import FileViewer
from supervisely.app.widgets.switch.switch import Switch
from supervisely.app.widgets.folder_thumbnail.folder_thumbnail import FolderThumbnail
from supervisely.app.widgets.file_thumbnail.file_thumbnail import FileThumbnail
from supervisely.app.widgets.model_info.model_info import ModelInfo
from supervisely.app.widgets.match_tags_or_classes.match_tags_or_classes import (
    MatchTagMetas,
    MatchObjClasses,
)
from supervisely.app.widgets.match_datasets.match_datasets import MatchDatasets
from supervisely.app.widgets.line_plot.line_plot import LinePlot
from supervisely.app.widgets.grid_plot.grid_plot import GridPlot
from supervisely.app.widgets.binded_input_number.binded_input_number import BindedInputNumber
from supervisely.app.widgets.augmentations.augmentations import Augmentations, AugmentationsWithTabs

from supervisely.app.widgets.tabs_dynamic.tabs_dynamic import TabsDynamic
from supervisely.app.widgets.stepper.stepper import Stepper
from supervisely.app.widgets.slider.slider import Slider
from supervisely.app.widgets.copy_to_clipboard.copy_to_clipboard import CopyToClipboard
from supervisely.app.widgets.file_storage_upload.file_storage_upload import FileStorageUpload
from supervisely.app.widgets.image_region_selector.image_region_selector import ImageRegionSelector
from supervisely.app.widgets.collapse.collapse import Collapse
from supervisely.app.widgets.team_files_selector.team_files_selector import TeamFilesSelector
from supervisely.app.widgets.icons.icons import Icons
from supervisely.app.widgets.badge.badge import Badge
from supervisely.app.widgets.date_picker.date_picker import DatePicker
from supervisely.app.widgets.datetime_picker.datetime_picker import DateTimePicker
from supervisely.app.widgets.transfer.transfer import Transfer
from supervisely.app.widgets.task_logs.task_logs import TaskLogs
from supervisely.app.widgets.reloadable_area.reloadable_area import ReloadableArea
from supervisely.app.widgets.image_pair_sequence.image_pair_sequence import ImagePairSequence
from supervisely.app.widgets.markdown.markdown import Markdown
from supervisely.app.widgets.class_balance.class_balance import ClassBalance
from supervisely.app.widgets.image_slider.image_slider import ImageSlider
from supervisely.app.widgets.rate.rate import Rate
from supervisely.app.widgets.carousel.carousel import Carousel
from supervisely.app.widgets.dropdown.dropdown import Dropdown
from supervisely.app.widgets.pie_chart.pie_chart import PieChart
