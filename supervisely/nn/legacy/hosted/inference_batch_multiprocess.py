# coding: utf-8

import os
import queue
import time
from collections import namedtuple
from copy import deepcopy
from threading import Thread

import multiprocessing as mp

from supervisely import logger
from supervisely.annotation.annotation import Annotation
from supervisely.imaging import image as sly_image
from supervisely.nn.config import AlwaysPassingConfigValidator
from supervisely.project.project import Project, read_single_project, OpenMode
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.paths import TaskPaths
from supervisely.task.progress import report_inference_finished
from supervisely.nn.hosted.inference_modes import InferenceModeFactory
from supervisely.nn.hosted.inference_batch import determine_task_inference_mode_config
from supervisely.task.progress import Progress


InferenceRequest = namedtuple('InferenceRequest', ['ds_name', 'item_name', 'item_paths'])
InferenceResponse = namedtuple('InferenceResponse', ['ds_name', 'item_name', 'item_paths', 'ann_json', 'meta_json'])


def single_inference_process_fn(inference_initializer, inference_mode_config, in_project_meta_json, request_queue,
                                result_meta_queue, progress_queue, project):
    """Loads a separate model, processes requests from request_queue, results go to result_queue.

    None request signals the process to finish.
    """
    single_image_inference = inference_initializer()
    inference_mode = InferenceModeFactory.create(
        inference_mode_config, ProjectMeta.from_json(in_project_meta_json), single_image_inference)

    project_meta_sent = False
    req = ''
    while req is not None:
        req = request_queue.get()
        if req is not None:
            # Send the resulting project meta to the parent project to make sure we only write the meta JSON once.
            if not project_meta_sent:
                try:
                    result_meta_queue.put(inference_mode.out_meta.to_json(), block=False)
                except queue.Full:
                    pass
            project_meta_sent = True

            in_ann = Annotation.load_json_file(req.item_paths.ann_path, inference_mode.out_meta)
            ann = inference_mode.infer_annotate_image_file(req.item_paths.img_path, in_ann)
            out_dataset = project.datasets.get(req.ds_name)
            out_dataset.add_item_file(
                req.item_name, req.item_paths.img_path, ann=ann, _validate_item=False, _use_hardlink=True)
            progress_queue.put(1)


def progress_report_thread_fn(in_project, progress_queue):
    """Gets inference result annotations from the queue and writes them to the output dataset.
    None result signals the thread to finish.
    """

    progress_bar = Progress('Model applying: ', in_project.total_items)
    while True:
        resp = progress_queue.get()
        if resp is not None:
            progress_bar.iter_done_report()
        else:
            break


def populate_inference_requests_queue(in_project, inference_processes, request_queue):
    for in_dataset in in_project:
        for in_item_name in in_dataset:
            logger.trace('Will process image',
                         extra={'dataset_name': in_dataset.name, 'image_name': in_item_name})
            in_item_paths = in_dataset.get_item_paths(in_item_name)
            req = InferenceRequest(ds_name=in_dataset.name, item_name=in_item_name,
                                   item_paths=in_item_paths)
            while True:
                try:
                    # Set a finite timeout for the queue insertion attempt to prevent deadlocks if all of the
                    # inference processes die in the interim.
                    request_queue.put(req, timeout=0.1)
                    break
                except queue.Full:
                    # If any of the inference processes has died, stop populating the requests queue and exit right
                    # away. Otherwise a deadlock is possible if no inference processes survive to take requests off
                    # the queue.
                    if not all(p.is_alive() for p in inference_processes):
                        # Early exit, return False to indicate failure.
                        return False
    return True


class BatchInferenceMultiprocessApplier:
    QUEUE_ELEMENTS_PER_PROCESS = 40

    def __init__(self, single_image_inference_initializer, num_processes, default_inference_mode_config: dict,
                 config_validator=None):
        self._config_validator = config_validator or AlwaysPassingConfigValidator()
        self._inference_mode_config = determine_task_inference_mode_config(deepcopy(default_inference_mode_config))

        self._in_project = read_single_project(TaskPaths.DATA_DIR)
        logger.info('Project structure has been read. Samples: {}.'.format(self._in_project.total_items))

        out_dir = os.path.join(TaskPaths.RESULTS_DIR, self._in_project.name)
        self._out_project = Project(out_dir, OpenMode.CREATE)
        # Create the output datasets in advance so that the worker processes do not collide on creating their own on
        # disk.
        for in_dataset in self._in_project:
            self._out_project.create_dataset(in_dataset.name)

        self._inference_request_queue = mp.Queue(maxsize=self.QUEUE_ELEMENTS_PER_PROCESS * num_processes)
        self._result_meta_queue = mp.Queue(maxsize=3)
        self._progress_report_queue = mp.Queue(maxsize=self.QUEUE_ELEMENTS_PER_PROCESS * num_processes)
        self._inference_processes = [
            mp.Process(
                target=single_inference_process_fn,
                args=(single_image_inference_initializer, self._inference_mode_config, self._in_project.meta.to_json(),
                      self._inference_request_queue, self._result_meta_queue,
                      self._progress_report_queue, self._out_project),
                daemon=True)
            for _ in range(num_processes)]
        logger.info('Dataset inference preparation done.')
        for p in self._inference_processes:
            p.start()

    def run_inference(self):
        progress_report_thread = Thread(
            target=progress_report_thread_fn, args=(self._in_project, self._progress_report_queue), daemon=True)
        progress_report_thread.start()

        feed_status = populate_inference_requests_queue(
            self._in_project, self._inference_processes, self._inference_request_queue)
        for _ in self._inference_processes:
            self._inference_request_queue.put(None)

        out_meta_json = self._result_meta_queue.get()
        self._out_project.set_meta(ProjectMeta.from_json(out_meta_json))

        for p in self._inference_processes:
            p.join()

        if not feed_status or not all(p.exitcode == 0 for p in self._inference_processes):
            raise RuntimeError('One of the inference processes encountered an error.')

        self._progress_report_queue.put(None)
        progress_report_thread.join()
        report_inference_finished()
