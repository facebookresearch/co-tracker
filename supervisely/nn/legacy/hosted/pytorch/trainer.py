from supervisely import logger
from supervisely.nn.dataset import ensure_samples_nonempty
from supervisely.nn.hosted.class_indexing import CONTINUE_TRAINING, TRANSFER_LEARNING
from supervisely.nn.hosted.pytorch.constants import CUSTOM_MODEL_CONFIG, HEAD_LAYER
from supervisely.nn.hosted.trainer import SuperviselyModelTrainer, BATCH_SIZE, DATASET_TAGS, EPOCHS, LOSS, LR, \
    TRAIN, VAL, WEIGHTS_INIT_TYPE, INPUT_SIZE, HEIGHT, WIDTH
from supervisely.nn.training.eval_planner import EvalPlanner, VAL_EVERY
from supervisely.nn.pytorch.dataset import PytorchSegmentationSlyDataset
from supervisely.nn.pytorch.weights import WeightsRW
from supervisely.task.paths import TaskPaths
from supervisely.task.progress import Progress, epoch_float, report_metrics_training, report_metrics_validation

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


def _check_all_pixels_have_segmentation_class(targets):
    if targets.min().item() < 0:
        raise ValueError('Training data has an item that is not fully covered by the segmentation labeling '
                         '(some image pixels are not assigned any class). This is an error. If you need to '
                         'ignore some parts of the input images, use a special dummy class and a custom loss '
                         'function that would ignore pixels of that class.')


class PytorchSegmentationTrainer(SuperviselyModelTrainer):
    @staticmethod
    def get_default_config():
        return {
            INPUT_SIZE: {
                WIDTH: 256,
                HEIGHT: 256
            },
            DATASET_TAGS: {
                TRAIN: 'train',
                VAL: 'val',
            },
            BATCH_SIZE: {
                TRAIN: 1,
                VAL: 1,
            },
            EPOCHS: 2,
            VAL_EVERY: 0.5,
            LR: 0.001,
            WEIGHTS_INIT_TYPE: TRANSFER_LEARNING,  # CONTINUE_TRAINING,
            CUSTOM_MODEL_CONFIG: {}    # Model-specific settings go in this section.
        }

    def __init__(self, model_factory_fn, optimization_loss_fn, training_metrics_dict=None, default_model_config=None):
        default_config = PytorchSegmentationTrainer.get_default_config()
        default_config[CUSTOM_MODEL_CONFIG].update(default_model_config if default_model_config is not None else {})

        self._model_factory_fn = model_factory_fn
        self._optimization_loss_fn = optimization_loss_fn
        self._training_metrics_dict = training_metrics_dict.copy() if training_metrics_dict is not None else {}
        if LOSS in self._training_metrics_dict:
            raise ValueError('aaaaaaa')
        self._metrics_with_loss = self._training_metrics_dict.copy()
        self._metrics_with_loss[LOSS] = self._optimization_loss_fn

        super().__init__(default_config=default_config)

    def get_start_class_id(self):
        """Set the integer segmentation class indices to start from 0.

        Some segmentation neural net implementations treat class id 0 in a special way (e.g. as a background class) and
        need the indexing to start from 1.
        """
        return 0

    def _determine_model_classes(self):
        """Look at input dataset segmentation classes and assign integer class ids."""

        # This also automatically reuses an existing class id mapping in continue_training mode (continuing training
        # from a previous snapshot with a dataset fully compatible in terms of classes).
        super()._determine_model_classes_segmentation(bkg_input_idx=-1)

    def _construct_and_fill_model(self):
        # Progress reporting to show a progress bar in the UI.
        model_build_progress = Progress('Building model:', 1)

        # Check the class name --> index mapping to infer the number of model output dimensions.
        num_classes = max(self.class_title_to_idx.values()) + 1

        # Initialize the model.
        model = self._model_factory_fn(
            num_classes=num_classes,
            input_size=self._input_size,
            custom_model_config=self.config.get(CUSTOM_MODEL_CONFIG, {}))
        logger.info('Model has been instantiated.')

        # Load model weights appropriate for the given training mode.
        weights_rw = WeightsRW(TaskPaths.MODEL_DIR)
        weights_init_type = self.config[WEIGHTS_INIT_TYPE]
        if weights_init_type == TRANSFER_LEARNING:
            # For transfer learning, do not attempt to load the weights for the model head. The existing snapshot may
            # have been trained on a different dataset, even on a different set of classes, and is in general not
            # compatible with the current model even in terms of dimensions. The head of the model will be initialized
            # randomly.
            self._model = weights_rw.load_for_transfer_learning(model, ignore_matching_layers=[HEAD_LAYER],
                                                                logger=logger)
        elif weights_init_type == CONTINUE_TRAINING:
            # Continuing training from an older snapshot requires full compatibility between the two models, including
            # class index mapping. Hence the snapshot weights must exactly match the structure of our model instance.
            self._model = weights_rw.load_strictly(model)

        # Model weights have been loaded, move them over to the GPU.
        self._model.cuda()

        # Advance the progress bar and log a progress message.
        logger.info('Weights have been loaded.', extra={WEIGHTS_INIT_TYPE: weights_init_type})
        model_build_progress.iter_done_report()

    def _construct_loss(self):
        pass

    def _construct_data_loaders(self):
        # Initialize the IO logic to feed the model during training.

        # Dimensionality of all images in an input batch must be the same.
        # We fix the input size for the whole dataset. Every image will be scaled to this size before feeding the model.
        src_size = self.config[INPUT_SIZE]
        input_size = (src_size[HEIGHT], src_size[WIDTH])

        # We need separate data loaders for the training and validation folds.
        self._data_loaders = {}

        # The train dataset should be re-shuffled every epoch, but the validation dataset samples order is fixed.
        for dataset_name, need_shuffle, drop_last in [(TRAIN, True, True), (VAL, False, False)]:
            # For more informative logging, grab the tag marking the respective dataset images.
            dataset_tag = self.config[DATASET_TAGS][dataset_name]

            # Get a list of samples for the dataset in question, make sure it is not empty.
            samples = self._samples_by_data_purpose[dataset_name]
            ensure_samples_nonempty(samples, dataset_tag, self.project.meta)

            # Instantiate the dataset object to handle sample indexing and image resizing.
            dataset = PytorchSegmentationSlyDataset(
                project_meta=self.project.meta,
                samples=samples,
                out_size=input_size,
                class_mapping=self.class_title_to_idx,
                bkg_color=-1
            )
            # Report progress.
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': dataset_name, 'tag': dataset_tag, 'samples': len(samples)
            })
            # Initialize a PyTorch data loader. For the training dataset, set the loader to ignore the last incomplete
            # batch to avoid noisy gradients and batchnorm updates.
            self._data_loaders[dataset_name] = DataLoader(
                dataset=dataset,
                batch_size=self.config[BATCH_SIZE][dataset_name],
                shuffle=need_shuffle,
                drop_last=drop_last
            )

        # Report progress
        logger.info('DataLoaders have been constructed.')

        # Compute the number of iterations per epoch for training and validation.
        self._train_iters = len(self._data_loaders[TRAIN])
        self._val_iters = len(self._data_loaders[VAL])
        self._epochs = self.config[EPOCHS]

        # Initialize a helper to determine when to pause training and perform validation and snapshotting.
        self._eval_planner = EvalPlanner(epochs=self._epochs, val_every=self.config[VAL_EVERY])

    def _dump_model_weights(self, out_dir):
        # Framework-specific logic to snapshot the model weights.
        WeightsRW(out_dir).save(self._model)

    def _validation(self):
        # Compute validation metrics.

        # Switch the model to evaluation model to stop batchnorm runnning average updates.
        self._model.eval()
        # Initialize the totals counters.
        validated_samples = 0
        total_val_metrics = {name: 0.0 for name in self._metrics_with_loss}
        total_loss = 0.0

        # Iterate over validation dataset batches.
        for val_it, (inputs, targets) in enumerate(self._data_loaders[VAL]):
            _check_all_pixels_have_segmentation_class(targets)

            # Move the data to the GPU and run inference.
            with torch.no_grad():
                inputs_cuda, targets_cuda = Variable(inputs).cuda(), Variable(targets).cuda()

            outputs_cuda = self._model(inputs_cuda)

            # The last betch may be smaller than the rest if the dataset does not have a whole number of full batches,
            # so read the batch size from the input.
            batch_size = inputs_cuda.size(0)

            # Compute the metrics and grab the values from GPU.
            batch_metrics = {name: metric_fn(outputs_cuda, targets_cuda).item()
                             for name, metric_fn in self._metrics_with_loss.items()}
            for name, metric_value in batch_metrics.items():
                total_val_metrics[name] += metric_value * batch_size

            # Add up the totals.
            validated_samples += batch_size

            # Report progress.
            logger.info("Validation in progress",
                        extra={'epoch': self.epoch_flt, 'val_iter': val_it, 'val_iters': self._val_iters})

        # Compute the average loss from the accumulated totals.
        avg_metrics_values = {name: total_value / validated_samples for name, total_value in total_val_metrics.items()}

        # Report progress and metric values to be plotted in the training chart and return.
        report_metrics_validation(self.epoch_flt, avg_metrics_values)
        logger.info("Validation has been finished", extra={'epoch': self.epoch_flt})
        return avg_metrics_values

    def train(self):
        # Initialize the progesss bar in the UI.
        training_progress = Progress('Model training: ', self._epochs * self._train_iters)

        # Initialize the optimizer.
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config[LR])
        # Running best loss value to determine which snapshot is the best so far.
        best_val_loss = float('inf')

        for epoch in range(self._epochs):
            logger.info("Starting new epoch", extra={'epoch': self.epoch_flt})
            for train_it, (inputs_cpu, targets_cpu) in enumerate(self._data_loaders[TRAIN]):
                _check_all_pixels_have_segmentation_class(targets_cpu)

                # Switch the model into training mode to enable gradient backpropagation and batch norm running average
                # updates.
                self._model.train()

                # Copy input batch to the GPU, run inference and compute optimization loss.
                inputs_cuda, targets_cuda = Variable(inputs_cpu).cuda(), Variable(targets_cpu).cuda()
                outputs_cuda = self._model(inputs_cuda)
                loss = self._optimization_loss_fn(outputs_cuda, targets_cuda)

                # Make a gradient descent step.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metric_values = {name: metric_fn(outputs_cuda, targets_cuda).item()
                                 for name, metric_fn in self._training_metrics_dict.items()}
                metric_values[LOSS] = loss.item()

                # Advance UI progess bar.
                training_progress.iter_done_report()
                # Compute fractional epoch value for more precise metrics reporting.
                self.epoch_flt = epoch_float(epoch, train_it + 1, self._train_iters)
                # Report metrics to be plotted in the training chart.
                report_metrics_training(self.epoch_flt, metric_values)

                # If needed, do validation and snapshotting.
                if self._eval_planner.need_validation(self.epoch_flt):
                    # Compute metrics on the validation dataset.
                    metrics_values_val = self._validation()

                    # Report progress.
                    self._eval_planner.validation_performed()

                    # Check whether the new weights are the best so far on the validation dataset.
                    val_loss = metrics_values_val[LOSS]
                    model_is_best = val_loss < best_val_loss
                    if model_is_best:
                        best_val_loss = val_loss

                    # Save a snapshot with the current weights. Mark whether the snapshot is the best so far in terms of
                    # validation loss.
                    self._save_model_snapshot(model_is_best, opt_data={
                        'epoch': self.epoch_flt,
                        'val_metrics': metrics_values_val,
                    })

            # Report progress
            logger.info("Epoch has finished", extra={'epoch': self.epoch_flt})
