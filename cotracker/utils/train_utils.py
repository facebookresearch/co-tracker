# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import signal
import socket
from torch.utils.data import ConcatDataset
from cotracker.datasets.utils import collate_fn, collate_fn_train
from torch.utils.tensorboard import SummaryWriter
from cotracker.datasets.dr_dataset import DynamicReplicaDataset
from cotracker.models.evaluation_predictor import EvaluationPredictor


# define the handler function
# for training on a slurm cluster
def sig_handler(signum, frame):
    print("caught signal", signum)
    print(socket.gethostname(), "USR1 signal caught.")
    # do other stuff to cleanup here
    print("requeuing job " + os.environ["SLURM_JOB_ID"])
    os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    sys.exit(-1)


def term_handler(signum, frame):
    print("bypassing sigterm", flush=True)


def get_eval_dataloader(dataset_root, ds_name):
    from cotracker.datasets.tap_vid_datasets import TapVidDataset

    collate_fn_local = collate_fn
    if ds_name == "dynamic_replica":
        from cotracker.datasets.dr_dataset import DynamicReplicaDataset

        eval_dataset = DynamicReplicaDataset(
            root=os.path.join(dataset_root, "dynamic_replica"),
            sample_len=300,
            only_first_n_samples=1,
            rgbd_input=False,
        )
    elif ds_name == "tapvid_davis_first":
        data_root = os.path.join(dataset_root, "tapvid/tapvid_davis/tapvid_davis.pkl")
        eval_dataset = TapVidDataset(
            dataset_type="davis", data_root=data_root, queried_first=True
        )
    elif ds_name == "tapvid_davis_strided":
        data_root = os.path.join(dataset_root, "tapvid/tapvid_davis/tapvid_davis.pkl")
        eval_dataset = TapVidDataset(
            dataset_type="davis", data_root=data_root, queried_first=False
        )
    elif ds_name == "tapvid_kinetics_first":
        eval_dataset = TapVidDataset(
            dataset_type="kinetics",
            data_root=os.path.join(dataset_root, "tapvid", "tapvid_kinetics"),
        )
    elif ds_name == "tapvid_stacking":
        eval_dataset = TapVidDataset(
            dataset_type="stacking",
            data_root=os.path.join(
                dataset_root, "tapvid", "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl"
            ),
        )
    elif ds_name == "tapvid_robotap":
        eval_dataset = TapVidDataset(
            dataset_type="robotap",
            data_root=os.path.join(dataset_root, "tapvid", "tapvid_robotap"),
        )
    elif ds_name == "kubric":
        from cotracker.datasets.kubric_movif_dataset import KubricMovifDataset

        eval_dataset = KubricMovifDataset(
            data_root=os.path.join(
                args.dataset_root, "kubric/kubric_movi_f_120_frames_dense/movi_f"
            ),
            traj_per_sample=1024,
            use_augs=False,
            split="valid",
            sample_vis_1st_frame=True,
        )
        collate_fn_local = collate_fn_train
    eval_dataloader_dr = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn_local,
    )
    return eval_dataloader_dr


def get_train_dataset(args):
    dataset = None
    if "kubric" in args.train_datasets:
        from cotracker.datasets import kubric_movif_dataset

        kubric = kubric_movif_dataset.KubricMovifDataset(
            data_root=os.path.join(
                args.dataset_root, "kubric/kubric_movi_f_120_frames_dense/movi_f"
            ),
            crop_size=args.crop_size,
            seq_len=args.sequence_len,
            traj_per_sample=args.traj_per_sample,
            sample_vis_last_frame=args.query_sampling_method is not None
            and ("random" in args.query_sampling_method),
            use_augs=not args.dont_use_augs,
            random_seq_len=args.random_seq_len,
            random_frame_rate=args.random_frame_rate,
            random_number_traj=args.random_number_traj,
        )

        if dataset is None:
            dataset = ConcatDataset(4 * [kubric])
        else:
            dataset = ConcatDataset(4 * [kubric] + [dataset])
        print("add kubric to train", len(dataset))

    if "dr" in args.train_datasets:
        dr = DynamicReplicaDataset(
            root=os.path.join(args.dataset_root, "dynamic_replica"),
            sample_len=args.sequence_len,
            split="train",
            traj_per_sample=args.traj_per_sample,
            crop_size=args.crop_size,
        )
        if dataset is None:
            dataset = dr
        else:
            dataset = ConcatDataset([dr] + [dataset])

    return dataset


def run_test_eval(evaluator, model, dataloaders, writer, step, query_random=False):
    model.eval()
    for ds_name, dataloader in dataloaders:
        visualize_every = 1
        grid_size = 5
        num_uniformly_sampled_pts = 0
        if ds_name == "dynamic_replica":
            visualize_every = 8
            grid_size = 0
        elif ds_name == "kubric":
            visualize_every = 5
            grid_size = 0
        elif "davis" in ds_name or "tapvid_stacking" in ds_name:
            visualize_every = 5
        elif "robotap" in ds_name:
            visualize_every = 20
        elif "kinetics" in ds_name:
            visualize_every = 50
        if query_random:
            grid_size = 0
            num_uniformly_sampled_pts = 100

        predictor = EvaluationPredictor(
            model.module.module,
            grid_size=grid_size,
            local_grid_size=0,
            single_point=False,
            num_uniformly_sampled_pts=num_uniformly_sampled_pts,
            n_iters=6,
        )

        if torch.cuda.is_available():
            predictor.model = predictor.model.cuda()

        metrics = evaluator.evaluate_sequence(
            model=predictor,
            test_dataloader=dataloader,
            dataset_name=ds_name,
            train_mode=True,
            writer=writer,
            step=step,
            visualize_every=visualize_every,
        )

        if ds_name == "dynamic_replica" or ds_name == "kubric":
            metrics = {
                f"{ds_name}_avg_{k}": v
                for k, v in metrics["avg"].items()
                if not ("1" in k or "2" in k or "4" in k or "8" in k)
            }

        if "tapvid" in ds_name:
            metrics = {
                f"{ds_name}_avg_OA": metrics["avg"]["occlusion_accuracy"],
                f"{ds_name}_avg_delta": metrics["avg"]["average_pts_within_thresh"],
                f"{ds_name}_avg_Jaccard": metrics["avg"]["average_jaccard"],
            }

        writer.add_scalars(f"Eval_{ds_name}", metrics, step)


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, ckpt_path):
        self.model = model
        self.scheduler = scheduler
        self.ckpt_path = ckpt_path
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=os.path.join(ckpt_path, "runs"))

    def _print_training_status(self):
        metrics_data = [
            self.running_loss[k] / Logger.SUM_FREQ
            for k in sorted(self.running_loss.keys())
        ]
        training_str = "[{:6d}] ".format(self.total_steps + 1)
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(
            f"Training Metrics ({self.total_steps}): {training_str + metrics_str}"
        )

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "runs"))

        for k in self.running_loss:
            self.writer.add_scalar(
                k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps
            )
            self.running_loss[k] = 0.0

    def push(self, metrics, task):
        self.total_steps += 1

        for key in metrics:
            task_key = str(key) + "_" + task
            if task_key not in self.running_loss:
                self.running_loss[task_key] = 0.0

            self.running_loss[task_key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "runs"))

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
