# coding: utf-8

import os
import os.path
import json
from copy import deepcopy

from supervisely import logger
from supervisely.io import fs as sly_fs
from supervisely.nn.config import update_recursively
from supervisely.nn.dataset import samples_by_tags
from supervisely.nn.hosted import class_indexing
from supervisely.nn.hosted.constants import SETTINGS, INPUT_SIZE, HEIGHT, WIDTH
from supervisely.task.progress import report_checkpoint_saved
from supervisely.project.project import read_single_project
from supervisely.geometry.rectangle import Rectangle
from supervisely.task.paths import TaskPaths
from supervisely.io.json import dump_json_file


BATCH_SIZE = 'batch_size'
LR = 'lr'
DATASET_TAGS = 'dataset_tags'
EPOCHS = 'epochs'
LOSS = 'loss'
TRAIN = 'train'
VAL = 'val'
WEIGHTS_INIT_TYPE = 'weights_init_type'


class TrainCheckpoints:
    def __init__(self, base_out_dir):
        self._base_out_dir = base_out_dir
        # Checkpoint index does not correspond to epoch. Depending on training config, checkpoints may be saved more
        # frequently than once per epoch (or less frequently than once per epoch).
        self._idx = -1
        self._prepare_next_dir()
        self._last_saved_dir = None

    def _prepare_next_dir(self):
        self._idx += 1
        self._subdir = '{:08}'.format(self._idx)
        self._odir = os.path.join(self._base_out_dir, self._subdir)
        sly_fs.mkdir(self._odir)

    def saved(self, is_best, sizeb, optional_data=None):
        report_best = is_best or (self._idx == 0)
        report_checkpoint_saved(checkpoint_idx=self._idx, subdir=self._subdir, sizeb=sizeb, best_now=report_best,
                                optional_data=optional_data)
        self._last_saved_dir = self._odir
        self._prepare_next_dir()

    def get_dir_to_write(self):
        return self._odir

    def get_last_ckpt_dir(self):
        return self._last_saved_dir


class SuperviselyModelTrainer:
    """
    Base class for train neural networks with Supervisely.

    It is highly recommended that your train classes subclass this class.

    Args:
        default_config: Dict object containing default training config.
    """
    def __init__(self, default_config):
        logger.info('Will init all required to train.')

        with open(TaskPaths.TASK_CONFIG_PATH) as fin:
            self._task_config = json.load(fin)

        self.checkpoints_saver = TrainCheckpoints(TaskPaths.RESULTS_DIR)
        self.project = read_single_project(TaskPaths.DATA_DIR)
        logger.info('Project structure has been read. Samples: {}.'.format(self.project.total_items))

        self._default_config = deepcopy(default_config)
        self._determine_config()
        self._determine_model_classes()
        self._determine_out_config()
        self._construct_samples_dct()
        self._construct_data_loaders()
        self._construct_and_fill_model()
        self._construct_loss()

        self.epoch_flt = 0

    @property
    def class_title_to_idx_key(self):
        return 'class_title_to_idx'

    @property
    def train_classes_key(self):
        return 'classes'

    def get_start_class_id(self):
        # Returns the first integer id to use when assigning integer ids to class names.
        # The usual setting for segmentation network, where background is often a special class with id 0.
        return 1

    def _validate_train_cfg(self, config):
        pass

    def _determine_config(self):
        """
        Determines train config by updating default config with user config and then validates them with given
        json schema.
        """
        input_config = self._task_config
        logger.info('Input config', extra={'config': input_config})
        config = deepcopy(self._default_config)
        update_recursively(config, input_config)
        logger.info('Full config', extra={'config': config})

        self._validate_train_cfg(config)
        self.config = config
        input_size_dict = self.config[INPUT_SIZE]
        self._input_size = (input_size_dict[HEIGHT], input_size_dict[WIDTH])

    def _determine_model_classes_segmentation(self, bkg_input_idx):
        self.out_classes, self.class_title_to_idx = class_indexing.create_segmentation_classes(
            in_project_classes=self.project.meta.obj_classes,
            special_classes_config=self.config.get('special_classes', {}),
            bkg_input_idx=bkg_input_idx,
            weights_init_type=self.config[WEIGHTS_INIT_TYPE],
            model_config_fpath=TaskPaths.MODEL_CONFIG_PATH,
            class_to_idx_config_key=self.class_title_to_idx_key,
            start_class_id=self.get_start_class_id())

    def _determine_model_classes_detection(self):
        in_project_classes = self.project.meta.obj_classes
        self.out_classes = class_indexing.make_out_classes(in_project_classes, geometry_type=Rectangle)
        logger.info('Determined model out classes', extra={'classes': list(self.out_classes)})
        in_project_class_to_idx = class_indexing.make_new_class_to_idx_map(
            in_project_classes, start_class_id=self.get_start_class_id())
        self.class_title_to_idx = class_indexing.infer_training_class_to_idx_map(
            self.config[WEIGHTS_INIT_TYPE],
            in_project_class_to_idx,
            TaskPaths.MODEL_CONFIG_PATH,
            class_to_idx_config_key=self.class_title_to_idx_key)
        logger.info('Determined class mapping.', extra={'class_mapping': self.class_title_to_idx})

    def _determine_model_classes(self):
        # Use _determine_model_classes_segmentation() or _determine_model_classes_detection() here depending on the
        # model needs.
        raise NotImplementedError()

    def _determine_out_config(self):
        self.out_config = {
            SETTINGS: self.config,
            self.train_classes_key: self.out_classes.to_json(),
            self.class_title_to_idx_key: self.class_title_to_idx,
        }

    def _construct_and_fill_model(self):
        raise NotImplementedError()

    def _construct_loss(self):
        # Useful for Tensorflow based models.
        raise NotImplementedError()

    def _construct_samples_dct(self):
        logger.info('Will collect samples (image/annotation pairs).')
        self.name_to_tag = self.config[DATASET_TAGS]
        self._deprecated_samples_by_tag = samples_by_tags(required_tags=list(self.name_to_tag.values()), project=self.project)
        self._samples_by_data_purpose = {purpose: self._deprecated_samples_by_tag[tag] for purpose, tag in self.config[DATASET_TAGS].items()}

    def _construct_data_loaders(self):
        # Pipeline-specific code to set up data loading should go here.
        raise NotImplementedError()

    def _dump_model_weights(self, out_dir):
        raise NotImplementedError

    def _save_model_snapshot(self, is_best, opt_data):
        out_dir = self.checkpoints_saver.get_dir_to_write()
        dump_json_file(self.out_config, os.path.join(out_dir, TaskPaths.MODEL_CONFIG_NAME))
        self._dump_model_weights(out_dir)
        size_bytes = sly_fs.get_directory_size(out_dir)
        self.checkpoints_saver.saved(is_best, size_bytes, opt_data)

    def train(self):
        """
        Start train process.
        """
        raise NotImplementedError()
