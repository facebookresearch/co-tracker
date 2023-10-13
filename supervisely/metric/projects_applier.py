# coding: utf-8

import os

from supervisely.task.progress import Progress
from supervisely.sly_logger import logger
from supervisely.task.paths import TaskPaths
from supervisely.metric.metric_base import MetricsBase
from supervisely.project.project import Project, OpenMode
from supervisely.annotation.annotation import Annotation

PROJECT_GT = 'project_gt'
PROJECT_PRED = 'project_pred'


class MetricProjectsApplier:

    def __init__(self, metric: MetricsBase, config: dict):
        self._metric = metric
        self._confg = config
        self._project_gt = self._get_project_or_die(PROJECT_GT)
        # If a separate project with predictions is not set, use the ground truth project as a source of predictions
        # (with potentially different classes/tags defined by the class mapping).
        self._project_pred = (
            self._get_project_or_die(PROJECT_PRED) if (PROJECT_PRED in self._confg) else self._project_gt)
        self._check_projects_compatible_structure()

    def _get_project_or_die(self, project_field_name):
        project_name = self._confg.get(project_field_name)
        if project_name is None:
            raise RuntimeError('Field "{}" does not exist in config.'.format(project_field_name))
        return Project(os.path.join(TaskPaths.DATA_DIR, project_name), OpenMode.READ)

    def _check_projects_compatible_structure(self):
        if self._project_gt.datasets.keys() != self._project_pred.datasets.keys():  # Keys is sorted - ok
            raise RuntimeError('Projects must contain same datasets.')
        if self._project_gt.total_items != self._project_pred.total_items:
            raise RuntimeError('Projects must contain same number of samples.')
        for ds_gt in self._project_gt.datasets:
            ds_pred = self._project_pred.datasets.get(ds_gt.name)
            for sample_name in ds_gt:
                if not ds_pred.item_exists(sample_name):
                    raise RuntimeError('Projects must contain identically named samples in respective datasets. ' +
                                       'Ground truth project has sample {!r} in dataset {!r}, but prediction project ' +
                                       'does not.'.format(sample_name, ds_gt.name))

        logger.info('Projects structure has been read. Samples: {} per project.'.format(self._project_gt.total_items))

    @property
    def project_gt(self):
        return self._project_gt

    @property
    def project_pred(self):
        return self._project_pred

    def run_evaluation(self):
        progress = Progress('metric evaluation', self._project_gt.total_items)
        for ds_name in self._project_gt.datasets.keys():
            ds_gt = self._project_gt.datasets.get(ds_name)
            ds_pred = self._project_pred.datasets.get(ds_name)

            for sample_name in ds_gt:
                try:
                    ann_gt = Annotation.load_json_file(ds_gt.get_ann_path(sample_name), self._project_gt.meta)
                    ann_pred = Annotation.load_json_file(ds_pred.get_ann_path(sample_name), self._project_pred.meta)
                    self._metric.add_pair(ann_gt, ann_pred)
                except ValueError as e:
                    logger.warning('An error has occured ({}). Sample "{}" in dataset "{}" will be skipped'
                                   .format(str(e), sample_name, ds_gt.name))
                progress.iter_done_report()
