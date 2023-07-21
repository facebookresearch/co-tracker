# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import os
from typing import Optional
import torch
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from cotracker.datasets.utils import dataclass_to_cuda_
from cotracker.utils.visualizer import Visualizer
from cotracker.models.core.model_utils import reduce_masked_mean
from cotracker.evaluation.core.eval_utils import compute_tapvid_metrics

import logging


class Evaluator:
    """
    A class defining the CoTracker evaluator.
    """

    def __init__(self, exp_dir) -> None:
        # Visualization
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.visualization_filepaths = defaultdict(lambda: defaultdict(list))
        self.visualize_dir = os.path.join(exp_dir, "visualisations")

    def compute_metrics(self, metrics, sample, pred_trajectory, dataset_name):
        if isinstance(pred_trajectory, tuple):
            pred_trajectory, pred_visibility = pred_trajectory
        else:
            pred_visibility = None
        if dataset_name == "badja":
            sample.segmentation = (sample.segmentation > 0).float()
            *_, N, _ = sample.trajectory.shape
            accs = []
            accs_3px = []
            for s1 in range(1, sample.video.shape[1]):  # target frame
                for n in range(N):
                    vis = sample.visibility[0, s1, n]
                    if vis > 0:
                        coord_e = pred_trajectory[0, s1, n]  # 2
                        coord_g = sample.trajectory[0, s1, n]  # 2
                        dist = torch.sqrt(torch.sum((coord_e - coord_g) ** 2, dim=0))
                        area = torch.sum(sample.segmentation[0, s1])
                        # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                        thr = 0.2 * torch.sqrt(area)
                        # correct =
                        accs.append((dist < thr).float())
                        # print('thr',thr)
                        accs_3px.append((dist < 3.0).float())

            res = torch.mean(torch.stack(accs)) * 100.0
            res_3px = torch.mean(torch.stack(accs_3px)) * 100.0
            metrics[sample.seq_name[0]] = res.item()
            metrics[sample.seq_name[0] + "_accuracy"] = res_3px.item()
            print(metrics)
            print(
                "avg", np.mean([v for k, v in metrics.items() if "accuracy" not in k])
            )
            print(
                "avg acc 3px",
                np.mean([v for k, v in metrics.items() if "accuracy" in k]),
            )
        elif dataset_name == "fastcapture" or ("kubric" in dataset_name):
            *_, N, _ = sample.trajectory.shape
            accs = []
            for s1 in range(1, sample.video.shape[1]):  # target frame
                for n in range(N):
                    vis = sample.visibility[0, s1, n]
                    if vis > 0:
                        coord_e = pred_trajectory[0, s1, n]  # 2
                        coord_g = sample.trajectory[0, s1, n]  # 2
                        dist = torch.sqrt(torch.sum((coord_e - coord_g) ** 2, dim=0))
                        thr = 3
                        correct = (dist < thr).float()
                        accs.append(correct)

            res = torch.mean(torch.stack(accs)) * 100.0
            metrics[sample.seq_name[0] + "_accuracy"] = res.item()
            print(metrics)
            print("avg", np.mean([v for v in metrics.values()]))
        elif "tapvid" in dataset_name:
            B, T, N, D = sample.trajectory.shape
            traj = sample.trajectory.clone()
            thr = 0.9

            if pred_visibility is None:
                logging.warning("visibility is NONE")
                pred_visibility = torch.zeros_like(sample.visibility)

            if not pred_visibility.dtype == torch.bool:
                pred_visibility = pred_visibility > thr

            # pred_trajectory
            query_points = sample.query_points.clone().cpu().numpy()

            pred_visibility = pred_visibility[:, :, :N]
            pred_trajectory = pred_trajectory[:, :, :N]

            gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
            gt_occluded = (
                torch.logical_not(sample.visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )

            pred_occluded = (
                torch.logical_not(pred_visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )
            pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()

            out_metrics = compute_tapvid_metrics(
                query_points,
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                query_mode="strided" if "strided" in dataset_name else "first",
            )

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = np.mean(
                    [v[metric_name] for k, v in metrics.items() if k != "avg"]
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            print("metrics", out_metrics)
            print("avg", metrics["avg"])
        else:
            rgbs = sample.video
            trajs_g = sample.trajectory
            valids = sample.valid
            vis_g = sample.visibility

            B, S, C, H, W = rgbs.shape
            assert C == 3
            B, S, N, D = trajs_g.shape

            assert torch.sum(valids) == B * S * N

            vis_g = (torch.sum(vis_g, dim=1, keepdim=True) >= 4).float().repeat(1, S, 1)

            ate = torch.norm(pred_trajectory - trajs_g, dim=-1)  # B, S, N

            metrics["things_all"] = reduce_masked_mean(ate, valids).item()
            metrics["things_vis"] = reduce_masked_mean(ate, valids * vis_g).item()
            metrics["things_occ"] = reduce_masked_mean(
                ate, valids * (1.0 - vis_g)
            ).item()

    @torch.no_grad()
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
    ):
        metrics = {}

        vis = Visualizer(
            save_dir=self.exp_dir,
            fps=7,
        )

        for ind, sample in enumerate(tqdm(test_dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue
            if torch.cuda.is_available():
                dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            if (
                not train_mode
                and hasattr(model, "sequence_len")
                and (sample.visibility[:, : model.sequence_len].sum() == 0)
            ):
                print(f"skipping batch {ind}")
                continue

            if "tapvid" in dataset_name:
                queries = sample.query_points.clone().float()

                queries = torch.stack(
                    [
                        queries[:, :, 0],
                        queries[:, :, 2],
                        queries[:, :, 1],
                    ],
                    dim=2,
                ).to(device)
            else:
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory[:, 0, :, :1]),
                        sample.trajectory[:, 0],
                    ],
                    dim=2,
                ).to(device)

            pred_tracks = model(sample.video, queries)
            if "strided" in dataset_name:

                inv_video = sample.video.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

                pred_trj, pred_vsb = pred_tracks
                inv_pred_trj, inv_pred_vsb = model(inv_video, inv_queries)

                inv_pred_trj = inv_pred_trj.flip(1)
                inv_pred_vsb = inv_pred_vsb.flip(1)

                mask = pred_trj == 0

                pred_trj[mask] = inv_pred_trj[mask]
                pred_vsb[mask[:, :, :, 0]] = inv_pred_vsb[mask[:, :, :, 0]]

                pred_tracks = pred_trj, pred_vsb

            if dataset_name == "badja" or dataset_name == "fastcapture":
                seq_name = sample.seq_name[0]
            else:
                seq_name = str(ind)

            vis.visualize(
                sample.video,
                pred_tracks[0] if isinstance(pred_tracks, tuple) else pred_tracks,
                filename=dataset_name + "_" + seq_name,
                writer=writer,
                step=step,
            )

            self.compute_metrics(metrics, sample, pred_tracks, dataset_name)
        return metrics
