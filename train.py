# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import signal
import socket
import sys
import json

import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.lite import LightningLite

from cotracker.models.evaluation_predictor import EvaluationPredictor
from cotracker.models.core.cotracker.cotracker import CoTracker
from cotracker.utils.visualizer import Visualizer
from cotracker.datasets.tap_vid_datasets import TapVidDataset
from cotracker.datasets.badja_dataset import BadjaDataset
from cotracker.datasets.fast_capture_dataset import FastCaptureDataset
from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.datasets import kubric_movif_dataset
from cotracker.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
from cotracker.models.core.cotracker.losses import sequence_loss, balanced_ce_loss


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler


def forward_batch(batch, model, args, loss_fn=None, writer=None, step=0):
    rgbs = batch.video
    trajs_g = batch.trajectory
    vis_g = batch.visibility
    valids = batch.valid
    B, T, C, H, W = rgbs.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = rgbs.device

    __, first_positive_inds = torch.max(vis_g, dim=1)
    # We want to make sure that during training the model sees visible points
    # that it does not need to track just yet: they are visible but queried from a later frame
    N_rand = N // 4
    # inds of visible points in the 1st frame
    nonzero_inds = [torch.nonzero(vis_g[0, :, i]) for i in range(N)]
    rand_vis_inds = torch.cat(
        [
            nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
            for nonzero_row in nonzero_inds
        ],
        dim=1,
    )
    first_positive_inds = torch.cat(
        [rand_vis_inds[:, :N_rand], first_positive_inds[:, N_rand:]], dim=1
    )
    ind_array_ = torch.arange(T, device=device)
    ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
    assert torch.allclose(
        vis_g[ind_array_ == first_positive_inds[:, None, :]],
        torch.ones_like(vis_g),
    )
    assert torch.allclose(
        vis_g[ind_array_ == rand_vis_inds[:, None, :]], torch.ones_like(vis_g)
    )

    gather = torch.gather(
        trajs_g, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, 2)
    )
    xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)

    queries = torch.cat([first_positive_inds[:, :, None], xys], dim=2)

    predictions, __, visibility, train_data = model(
        rgbs=rgbs, queries=queries, iters=args.train_iters, is_train=True
    )
    vis_predictions, coord_predictions, wind_inds, sort_inds = train_data

    trajs_g = trajs_g[:, :, sort_inds]
    vis_g = vis_g[:, :, sort_inds]
    valids = valids[:, :, sort_inds]

    vis_gts = []
    traj_gts = []
    valids_gts = []

    for i, wind_idx in enumerate(wind_inds):
        ind = i * (args.sliding_window_len // 2)

        vis_gts.append(vis_g[:, ind : ind + args.sliding_window_len, :wind_idx])
        traj_gts.append(trajs_g[:, ind : ind + args.sliding_window_len, :wind_idx])
        valids_gts.append(valids[:, ind : ind + args.sliding_window_len, :wind_idx])

    seq_loss = sequence_loss(coord_predictions, traj_gts, vis_gts, valids_gts, 0.8)
    vis_loss = balanced_ce_loss(vis_predictions, vis_gts, valids_gts)

    output = {"flow": {"predictions": predictions[0].detach()}}
    output["flow"]["loss"] = seq_loss.mean()
    output["visibility"] = {
        "loss": vis_loss.mean() * 10.0,
        "predictions": visibility[0].detach(),
    }
    return output


def run_test_eval(evaluator, model, dataloaders, writer, step):
    model.eval()
    for ds_name, dataloader in dataloaders:
        predictor = EvaluationPredictor(
            model.module.module,
            grid_size=6,
            local_grid_size=0,
            single_point=False,
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
        )

        if ds_name == "badja" or ds_name == "fastcapture" or ("kubric" in ds_name):

            metrics = {
                **{
                    f"{ds_name}_avg": np.mean(
                        [v for k, v in metrics.items() if "accuracy" not in k]
                    )
                },
                **{
                    f"{ds_name}_avg_accuracy": np.mean(
                        [v for k, v in metrics.items() if "accuracy" in k]
                    )
                },
            }
            print("avg", np.mean([v for v in metrics.values()]))

        if "tapvid" in ds_name:
            metrics = {
                f"{ds_name}_avg_OA": metrics["avg"]["occlusion_accuracy"] * 100,
                f"{ds_name}_avg_delta": metrics["avg"]["average_pts_within_thresh"]
                * 100,
                f"{ds_name}_avg_Jaccard": metrics["avg"]["average_jaccard"] * 100,
            }

        writer.add_scalars(f"Eval", metrics, step)


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=os.path.join(args.ckpt_path, "runs"))

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
            self.writer = SummaryWriter(log_dir=os.path.join(args.ckpt_path, "runs"))

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
            self.writer = SummaryWriter(log_dir=os.path.join(args.ckpt_path, "runs"))

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


class Lite(LightningLite):
    def run(self, args):
        def seed_everything(seed: int):
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        eval_dataloaders = []
        if "badja" in args.eval_datasets:
            eval_dataset = BadjaDataset(
                data_root=os.path.join(args.dataset_root, "BADJA"),
                max_seq_len=args.eval_max_seq_len,
                dataset_resolution=args.crop_size,
            )
            eval_dataloader_badja = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=8,
                collate_fn=collate_fn,
            )
            eval_dataloaders.append(("badja", eval_dataloader_badja))

        if "fastcapture" in args.eval_datasets:
            eval_dataset = FastCaptureDataset(
                data_root=os.path.join(args.dataset_root, "fastcapture"),
                max_seq_len=min(100, args.eval_max_seq_len),
                max_num_points=40,
                dataset_resolution=args.crop_size,
            )
            eval_dataloader_fastcapture = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=collate_fn,
            )
            eval_dataloaders.append(("fastcapture", eval_dataloader_fastcapture))

        if "tapvid_davis_first" in args.eval_datasets:
            data_root = os.path.join(args.dataset_root, "tapvid_davis/tapvid_davis.pkl")
            eval_dataset = TapVidDataset(dataset_type="davis", data_root=data_root)
            eval_dataloader_tapvid_davis = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=collate_fn,
            )
            eval_dataloaders.append(("tapvid_davis", eval_dataloader_tapvid_davis))

        evaluator = Evaluator(args.ckpt_path)

        visualizer = Visualizer(
            save_dir=args.ckpt_path,
            pad_value=80,
            fps=1,
            show_first_frame=0,
            tracks_leave_trace=0,
        )

        loss_fn = None

        if args.model_name == "cotracker":

            model = CoTracker(
                stride=args.model_stride,
                S=args.sliding_window_len,
                add_space_attn=not args.remove_space_attn,
                num_heads=args.updateformer_num_heads,
                hidden_size=args.updateformer_hidden_size,
                space_depth=args.updateformer_space_depth,
                time_depth=args.updateformer_time_depth,
            )
        else:
            raise ValueError(f"Model {args.model_name} doesn't exist")

        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        model.cuda()

        train_dataset = kubric_movif_dataset.KubricMovifDataset(
            data_root=os.path.join(args.dataset_root, "kubric_movi_f"),
            crop_size=args.crop_size,
            seq_len=args.sequence_len,
            traj_per_sample=args.traj_per_sample,
            sample_vis_1st_frame=args.sample_vis_1st_frame,
            use_augs=not args.dont_use_augs,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
            collate_fn=collate_fn_train,
            drop_last=True,
        )

        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)
        print("LEN TRAIN LOADER", len(train_loader))
        optimizer, scheduler = fetch_optimizer(args, model)

        total_steps = 0
        logger = Logger(model, scheduler)

        folder_ckpts = [
            f
            for f in os.listdir(args.ckpt_path)
            if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f
        ]
        if len(folder_ckpts) > 0:
            ckpt_path = sorted(folder_ckpts)[-1]
            ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
            logging.info(f"Loading checkpoint {ckpt_path}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")

        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(
                ".pt"
            )
            logging.info("Loading checkpoint...")

            strict = True
            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict, strict=strict)

            logging.info(f"Done loading checkpoint")
        model, optimizer = self.setup(model, optimizer, move_to_device=False)
        # model.cuda()
        model.train()

        save_freq = args.save_freq
        scaler = GradScaler(enabled=args.mixed_precision)

        should_keep_training = True
        global_batch_num = 0
        epoch = -1

        while should_keep_training:
            epoch += 1
            for i_batch, batch in enumerate(tqdm(train_loader)):
                batch, gotit = batch
                if not all(gotit):
                    print("batch is None")
                    continue
                dataclass_to_cuda_(batch)

                optimizer.zero_grad()

                assert model.training

                output = forward_batch(
                    batch,
                    model,
                    args,
                    loss_fn=loss_fn,
                    writer=logger.writer,
                    step=total_steps,
                )

                loss = 0
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]
                        logger.writer.add_scalar(
                            f"live_{k}_loss", v["loss"].item(), total_steps
                        )
                    if "metrics" in v:
                        logger.push(v["metrics"], k)

                if self.global_rank == 0:
                    if total_steps % save_freq == save_freq - 1:
                        if args.model_name == "motion_diffuser":
                            pred_coords = model.module.module.forward_batch_test(
                                batch, interp_shape=args.crop_size
                            )

                            output["flow"] = {"predictions": pred_coords[0].detach()}
                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=batch.trajectory.clone(),
                            filename="train_gt_traj",
                            writer=logger.writer,
                            step=total_steps,
                        )

                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=output["flow"]["predictions"][None],
                            filename="train_pred_traj",
                            writer=logger.writer,
                            step=total_steps,
                        )

                    if len(output) > 1:
                        logger.writer.add_scalar(
                            f"live_total_loss", loss.item(), total_steps
                        )
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    global_batch_num += 1

                self.barrier()

                self.backward(scaler.scale(loss))

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                total_steps += 1
                if self.global_rank == 0:
                    if (i_batch >= len(train_loader) - 1) or (
                        total_steps == 1 and args.validate_at_start
                    ):
                        if (epoch + 1) % args.save_every_n_epoch == 0:
                            ckpt_iter = "0" * (6 - len(str(total_steps))) + str(
                                total_steps
                            )
                            save_path = Path(
                                f"{args.ckpt_path}/model_{args.model_name}_{ckpt_iter}.pth"
                            )

                            save_dict = {
                                "model": model.module.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "total_steps": total_steps,
                            }

                            logging.info(f"Saving file {save_path}")
                            self.save(save_dict, save_path)

                        if (epoch + 1) % args.evaluate_every_n_epoch == 0 or (
                            args.validate_at_start and epoch == 0
                        ):
                            run_test_eval(
                                evaluator,
                                model,
                                eval_dataloaders,
                                logger.writer,
                                total_steps,
                            )
                            model.train()
                            torch.cuda.empty_cache()

                self.barrier()
                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

        print("FINISHED TRAINING")

        PATH = f"{args.ckpt_path}/{args.model_name}_final.pth"
        torch.save(model.module.module.state_dict(), PATH)
        run_test_eval(evaluator, model, eval_dataloaders, logger.writer, total_steps)
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="cotracker", help="model name")
    parser.add_argument("--restore_ckpt", help="path to restore a checkpoint")
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size used during training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=6, help="number of dataloader workers"
    )

    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="max learning rate.")
    parser.add_argument(
        "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200000, help="length of training schedule."
    )
    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=1,
        help="evaluate during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--save_every_n_epoch",
        type=int,
        default=1,
        help="save checkpoints during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--validate_at_start",
        action="store_true",
        help="whether to run evaluation before training starts",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="frequency of trajectory visualization during training",
    )
    parser.add_argument(
        "--traj_per_sample",
        type=int,
        default=768,
        help="the number of trajectories to sample for training",
    )
    parser.add_argument(
        "--dataset_root", type=str, help="path lo all the datasets (train and eval)"
    )

    parser.add_argument(
        "--train_iters",
        type=int,
        default=4,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument(
        "--sequence_len", type=int, default=8, help="train sequence length"
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["things", "badja"],
        help="what datasets to use for evaluation",
    )

    parser.add_argument(
        "--remove_space_attn",
        action="store_true",
        help="remove space attention from CoTracker",
    )
    parser.add_argument(
        "--dont_use_augs",
        action="store_true",
        help="don't apply augmentations during training",
    )
    parser.add_argument(
        "--sample_vis_1st_frame",
        action="store_true",
        help="only sample trajectories with points visible on the first frame",
    )
    parser.add_argument(
        "--sliding_window_len",
        type=int,
        default=8,
        help="length of the CoTracker sliding window",
    )
    parser.add_argument(
        "--updateformer_hidden_size",
        type=int,
        default=384,
        help="hidden dimension of the CoTracker transformer model",
    )
    parser.add_argument(
        "--updateformer_num_heads",
        type=int,
        default=8,
        help="number of heads of the CoTracker transformer model",
    )
    parser.add_argument(
        "--updateformer_space_depth",
        type=int,
        default=12,
        help="number of group attention layers in the CoTracker transformer model",
    )
    parser.add_argument(
        "--updateformer_time_depth",
        type=int,
        default=12,
        help="number of time attention layers in the CoTracker transformer model",
    )
    parser.add_argument(
        "--model_stride",
        type=int,
        default=8,
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=[384, 512],
        help="crop videos to this resolution during training",
    )
    parser.add_argument(
        "--eval_max_seq_len",
        type=int,
        default=1000,
        help="maximum length of evaluation videos",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices="auto",
        accelerator="gpu",
        precision=32,
        # num_nodes=4,
    ).run(args)
