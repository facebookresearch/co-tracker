# coding: utf-8

from copy import deepcopy
import os

from supervisely import logger
from supervisely.annotation.annotation import Annotation
from supervisely.imaging import image as sly_image
from supervisely.io.json import load_json_file
from supervisely.nn.config import AlwaysPassingConfigValidator
from supervisely.project.project import Project, read_single_project, OpenMode
from supervisely.task.paths import TaskPaths
from supervisely.task.progress import report_inference_finished
from supervisely.nn.hosted.inference_single_image import SingleImageInferenceBase
from supervisely.nn.hosted.inference_modes import MODE, InferenceModeFactory, get_effective_inference_mode_config
from supervisely.nn.hosted.legacy.inference_config import maybe_convert_from_v1_inference_task_config
from supervisely.task.progress import Progress


def determine_task_inference_mode_config(default_inference_mode_config):
    raw_task_config = load_json_file(TaskPaths.TASK_CONFIG_PATH)
    task_config = maybe_convert_from_v1_inference_task_config(raw_task_config)
    logger.info('Input task config', extra={'config': task_config})
    result_config = get_effective_inference_mode_config(
        task_config.get(MODE, {}), default_inference_mode_config)
    logger.info('Full inference mode config', extra={'config': result_config})
    return result_config


class BatchInferenceApplier:
    """Runs a given single image inference model over all images in a project; saves results to a new project."""

    def __init__(self, single_image_inference: SingleImageInferenceBase, default_inference_mode_config: dict,
                 config_validator=None):
        self._single_image_inference = single_image_inference
        self._config_validator = config_validator or AlwaysPassingConfigValidator()

        self._inference_mode_config = determine_task_inference_mode_config(deepcopy(default_inference_mode_config))
        self._determine_input_data()
        logger.info('Dataset inference preparation done.')

    def _determine_input_data(self):
        # TODO support multiple input projects.
        self._in_project = read_single_project(TaskPaths.DATA_DIR)
        logger.info('Project structure has been read. Samples: {}.'.format(self._in_project.total_items))

    def run_inference(self):
        inference_mode = InferenceModeFactory.create(
            self._inference_mode_config, self._in_project.meta, self._single_image_inference)
        out_project = Project(os.path.join(TaskPaths.RESULTS_DIR, self._in_project.name), OpenMode.CREATE)
        out_project.set_meta(inference_mode.out_meta)

        progress_bar = Progress('Model applying: ', self._in_project.total_items)
        for in_dataset in self._in_project:
            out_dataset = out_project.create_dataset(in_dataset.name)
            for in_item_name in in_dataset:
                # Use output project meta so that we get an annotation that is already in the context of the output
                # project (with added object classes etc).
                in_item_paths = in_dataset.get_item_paths(in_item_name)
                in_ann = Annotation.load_json_file(in_item_paths.ann_path, inference_mode.out_meta)
                logger.trace('Will process image', extra={'dataset_name': in_dataset.name, 'image_name': in_item_name})
                inference_annotation = inference_mode.infer_annotate_image_file(in_item_paths.img_path, in_ann)
                out_dataset.add_item_file(in_item_name, in_item_paths.img_path, ann=inference_annotation,
                                          _validate_item=False, _use_hardlink=True)

                progress_bar.iter_done_report()

        report_inference_finished()
