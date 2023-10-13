# coding: utf-8
from copy import deepcopy
import json

from supervisely import logger
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.imaging.image import read as sly_image_read
from supervisely.io.json import load_json_file
from supervisely.nn.hosted.constants import MODEL, SETTINGS, INPUT_SIZE, HEIGHT, WIDTH
from supervisely.nn.hosted.legacy.inference_config import maybe_convert_from_v1_inference_task_config, \
                                                              maybe_convert_from_deploy_task_config
from supervisely.nn.config import update_recursively
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.paths import TaskPaths
from supervisely.task.progress import Progress
from supervisely.worker_api.interfaces import SingleImageInferenceInterface


GPU_DEVICE = 'gpu_device'


class SingleImageInferenceBase(SingleImageInferenceInterface):
    def __init__(self, task_model_config=None, _load_model_weights=True):
        logger.info('Starting base single image inference applier init.')
        task_model_config = self._load_task_model_config() if task_model_config is None else deepcopy(task_model_config)
        self._config = update_recursively(self.get_default_config(), task_model_config)
        # Only validate after merging task config with the defaults.
        self._validate_model_config(self._config)

        self._load_train_config()
        if _load_model_weights:
            self._construct_and_fill_model()
        logger.info('Base single image inference applier init done.')

    def _construct_and_fill_model(self):
        progress_dummy = Progress('Building model:', 1)
        progress_dummy.iter_done_report()

    def _validate_model_config(self, config):
        pass

    def inference(self, image, ann):
        raise NotImplementedError()

    def inference_image_file(self, image_file, ann):
        image = sly_image_read(image_file)
        return self.inference(image, ann)

    @staticmethod
    def get_default_config():
        return {}

    @property
    def class_title_to_idx_key(self):
        return 'class_title_to_idx'

    @property
    def train_classes_key(self):
        return 'classes'

    @property
    def model_out_meta(self):
        return self._model_out_meta

    def get_out_meta(self):
        return self._model_out_meta

    def _model_out_tags(self):
        return TagMetaCollection()  # Empty by default

    def _load_raw_model_config_json(self):
        try:
            with open(TaskPaths.MODEL_CONFIG_PATH) as fin:
                self.train_config = json.load(fin)
        except FileNotFoundError:
            raise RuntimeError('Unable to run inference, config from training was not found.')

    @staticmethod
    def _load_task_model_config():
        raw_task_config = load_json_file(TaskPaths.TASK_CONFIG_PATH)
        raw_task_config = maybe_convert_from_deploy_task_config(raw_task_config)
        task_config = maybe_convert_from_v1_inference_task_config(raw_task_config)
        return task_config[MODEL]

    def _load_train_config(self):
        self._load_raw_model_config_json()

        self.class_title_to_idx = self.train_config[self.class_title_to_idx_key]
        logger.info('Read model internal class mapping', extra={'class_mapping': self.class_title_to_idx})
        train_classes = ObjClassCollection.from_json(self.train_config[self.train_classes_key])
        logger.info('Read model out classes', extra={'classes': train_classes.to_json()})

        # TODO: Factor out meta constructing from _load_train_config method.
        self._model_out_meta = ProjectMeta(obj_classes=train_classes, tag_metas=self._model_out_tags())
        # Make a separate [index] --> [class] map that excludes the 'special' classes that should not be in the`
        # final output.
        self.out_class_mapping = {idx: train_classes.get(title) for title, idx in self.class_title_to_idx.items() if
                                  train_classes.has_key(title)}

    def _determine_model_input_size(self):
        src_size = self.train_config[SETTINGS][INPUT_SIZE]
        self.input_size = (src_size[HEIGHT], src_size[WIDTH])
        logger.info('Model input size is read (for auto-rescale).', extra={INPUT_SIZE: {
            WIDTH: self.input_size[1], HEIGHT: self.input_size[0]
        }})
