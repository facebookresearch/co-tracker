# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict

import hydra
import numpy as np

import torch
from omegaconf import OmegaConf

from cotracker.datasets.badja_dataset import BadjaDataset
from cotracker.datasets.fast_capture_dataset import FastCaptureDataset
from cotracker.datasets.tap_vid_datasets import TapVidDataset
from cotracker.datasets.utils import collate_fn

from cotracker.models.evaluation_predictor import EvaluationPredictor

from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.models.build_cotracker import (
    build_cotracker,
)


@dataclass(eq=False)
class DefaultConfig:
    exp_dir: str = "./outputs"
    exp_idx: int = 0

    dataset_name: str = "badja"

    # Model
    checkpoint: str = "./checkpoints/cotracker_stride_4_wind_8.pth"
    # cotracker_stride_4_wind_12
    # cotracker_stride_8_wind_16

    # EvaluationPredictor
    N_grid: int = 6
    N_local_grid: int = 6
    single_point: bool = True
    n_iters: int = 6

    seed: int = 0
    gpu_idx: int = 0

    # Override hydra's working directory to current working dir,
    # also disable storing the .hydra logs:
    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},
            "output_subdir": None,
        }
    )


def run_eval(cfg: DefaultConfig):
    """
    Evaluates new view synthesis metrics of a specified model
    on a benchmark dataset.
    """
    # make the experiment directory
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # dump the exp cofig to the exp_dir
    cfg_file = os.path.join(cfg.exp_dir, "expconfig.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    # writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, "runs"))
    evaluator = Evaluator(cfg.exp_dir)
    # from cotracker.models.pips_predictor import PIPsPredictor

    # cotracker_model = PIPsPredictor()
    # model_zoo(**cfg.MODEL)
    cotracker_model = build_cotracker(cfg.checkpoint)

    predictor = EvaluationPredictor(
        cotracker_model,
        N_grid=cfg.N_grid,
        N_local_grid=cfg.N_local_grid,
        single_point=cfg.single_point,
        n_iters=cfg.n_iters,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    curr_collate_fn = collate_fn
    if cfg.dataset_name == "badja":
        test_dataset = BadjaDataset("/checkpoint/nikitakaraev/2023_mimo/datasets/BADJA")
    elif cfg.dataset_name == "fastcapture":
        test_dataset = FastCaptureDataset(max_seq_len=100, max_num_points=20)
    elif "tapvid" in cfg.dataset_name:
        dataset_type = cfg.dataset_name.split("_")[1]
        if dataset_type == "davis":
            root_path = "/checkpoint/nikitakaraev/2023_mimo/datasets/tapvid_davis/tapvid_davis.pkl"
        elif dataset_type == "robotics":
            root_path = "/checkpoint/nikitakaraev/2023_mimo/datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl"
        elif dataset_type == "kinetics":
            root_path = "/checkpoint/nikitakaraev/2023_mimo/datasets/kinetics/kinetics-dataset/k700-2020/tapvid_kinetics"
        test_dataset = TapVidDataset(
            dataset_type=dataset_type,
            root_path=root_path,
            queried_first=not "strided" in cfg.dataset_name,
        )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=14,
        collate_fn=curr_collate_fn,
    )
    import time

    start = time.time()

    evaluate_result = evaluator.evaluate_sequence(
        predictor,
        test_dataloader,
        dataset_name=cfg.dataset_name,
    )

    end = time.time()
    print(end - start)
    if not "tapvid" in cfg.dataset_name:
        print("evaluate_result", evaluate_result)
    else:
        evaluate_result = evaluate_result["avg"]
    result_file = os.path.join(cfg.exp_dir, f"result_eval_.json")
    evaluate_result["time"] = end - start
    print(f"Dumping eval results to {result_file}.")
    with open(result_file, "w") as f:
        json.dump(evaluate_result, f)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config_eval", node=DefaultConfig)


@hydra.main(config_path="./configs/", config_name="default_config_eval")
def evaluate(cfg: DefaultConfig) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_idx)
    run_eval(cfg)


if __name__ == "__main__":
    evaluate()
