# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import hydra
import numpy as np
import torch

from typing import Optional
from dataclasses import dataclass, field

from omegaconf import OmegaConf

from cotracker.datasets.utils import collate_fn
from cotracker.models.evaluation_predictor import EvaluationPredictor

from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.models.build_cotracker import build_cotracker


@dataclass(eq=False)
class DefaultConfig:
    # Directory where all outputs of the experiment will be saved.
    exp_dir: str = "./outputs"

    # Name of the dataset to be used for the evaluation.
    dataset_name: str = "tapvid_davis_first"
    # The root directory of the dataset.
    dataset_root: str = "./"

    # Path to the pre-trained model checkpoint to be used for the evaluation.
    # The default value is the path to a specific CoTracker model checkpoint.
    checkpoint: str = "./checkpoints/scaled_online.pth"
    # EvaluationPredictor parameters
    # The size (N) of the support grid used in the predictor.
    # The total number of points is (N*N).
    grid_size: int = 5
    # The size (N) of the local support grid.
    local_grid_size: int = 8
    num_uniformly_sampled_pts: int = 0
    sift_size: int = 0
    # A flag indicating whether to evaluate one ground truth point at a time.
    single_point: bool = False
    offline_model: bool = False
    window_len: int = 16
    # The number of iterative updates for each sliding window.
    n_iters: int = 6

    seed: int = 0
    gpu_idx: int = 0
    local_extent: int = 50

    v2: bool = False

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
    The function evaluates CoTracker on a specified benchmark dataset based on a provided configuration.

    Args:
        cfg (DefaultConfig): An instance of DefaultConfig class which includes:
            - exp_dir (str): The directory path for the experiment.
            - dataset_name (str): The name of the dataset to be used.
            - dataset_root (str): The root directory of the dataset.
            - checkpoint (str): The path to the CoTracker model's checkpoint.
            - single_point (bool): A flag indicating whether to evaluate one ground truth point at a time.
            - n_iters (int): The number of iterative updates for each sliding window.
            - seed (int): The seed for setting the random state for reproducibility.
            - gpu_idx (int): The index of the GPU to be used.
    """
    # Creating the experiment directory if it doesn't exist
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # Saving the experiment configuration to a .yaml file in the experiment directory
    cfg_file = os.path.join(cfg.exp_dir, "expconfig.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    evaluator = Evaluator(cfg.exp_dir)
    cotracker_model = build_cotracker(
        cfg.checkpoint, offline=cfg.offline_model, window_len=cfg.window_len, v2=cfg.v2
    )

    # Creating the EvaluationPredictor object
    predictor = EvaluationPredictor(
        cotracker_model,
        grid_size=cfg.grid_size,
        local_grid_size=cfg.local_grid_size,
        sift_size=cfg.sift_size,
        single_point=cfg.single_point,
        num_uniformly_sampled_pts=cfg.num_uniformly_sampled_pts,
        n_iters=cfg.n_iters,
        local_extent=cfg.local_extent,
        interp_shape=(384, 512),
    )

    if torch.cuda.is_available():
        predictor.model = predictor.model.cuda()

    # Setting the random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Constructing the specified dataset
    curr_collate_fn = collate_fn
    if "tapvid" in cfg.dataset_name:
        from cotracker.datasets.tap_vid_datasets import TapVidDataset

        dataset_type = cfg.dataset_name.split("_")[1]
        if dataset_type == "davis":
            data_root = os.path.join(
                cfg.dataset_root, "tapvid_davis", "tapvid_davis.pkl"
            )
        elif dataset_type == "kinetics":
            data_root = os.path.join(cfg.dataset_root, "tapvid_kinetics")
        elif dataset_type == "robotap":
            data_root = os.path.join(cfg.dataset_root, "tapvid_robotap")
        elif dataset_type == "stacking":
            data_root = os.path.join(
                cfg.dataset_root, "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl"
            )

        test_dataset = TapVidDataset(
            dataset_type=dataset_type,
            data_root=data_root,
            queried_first=not "strided" in cfg.dataset_name,
            # resize_to=None,
        )
    elif cfg.dataset_name == "dynamic_replica":
        from cotracker.datasets.dr_dataset import DynamicReplicaDataset

        test_dataset = DynamicReplicaDataset(
            cfg.dataset_root, sample_len=300, only_first_n_samples=1
        )

    # Creating the DataLoader object
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=curr_collate_fn,
    )

    # Timing and conducting the evaluation
    import time

    start = time.time()
    evaluate_result = evaluator.evaluate_sequence(
        predictor, test_dataloader, dataset_name=cfg.dataset_name
    )
    end = time.time()
    print(end - start)

    # Saving the evaluation results to a .json file
    evaluate_result = evaluate_result["avg"]
    print("evaluate_result", evaluate_result)
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
