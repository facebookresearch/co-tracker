# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import gzip
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple

from cotracker.datasets.utils import CoTrackerData
from cotracker.datasets.dataclass_utils import load_dataclass


@dataclass
class ImageAnnotation:
    # path to jpg file, relative w.r.t. dataset_root
    path: str
    # H x W
    size: Tuple[int, int]


@dataclass
class DynamicReplicaFrameAnnotation:
    """A dataclass used to load annotations from json."""

    # can be used to join with `SequenceAnnotation`
    sequence_name: str
    # 0-based, continuous frame number within sequence
    frame_number: int
    # timestamp in seconds from the video start
    frame_timestamp: float

    image: ImageAnnotation
    meta: Optional[Dict[str, Any]] = None

    camera_name: Optional[str] = None
    trajectories: Optional[str] = None


class DynamicReplicaDataset(data.Dataset):
    def __init__(
        self,
        root,
        split="valid",
        traj_per_sample=256,
        crop_size=None,
        sample_len=-1,
        only_first_n_samples=-1,
        rgbd_input=False,
    ):
        super(DynamicReplicaDataset, self).__init__()
        self.root = root
        self.sample_len = sample_len
        self.split = split
        self.traj_per_sample = traj_per_sample
        self.rgbd_input = rgbd_input
        self.crop_size = crop_size
        frame_annotations_file = f"frame_annotations_{split}.jgz"
        self.sample_list = []
        with gzip.open(
            os.path.join(root, split, frame_annotations_file), "rt", encoding="utf8"
        ) as zipfile:
            frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])
        seq_annot = defaultdict(list)
        for frame_annot in frame_annots_list:
            if frame_annot.camera_name == "left":
                seq_annot[frame_annot.sequence_name].append(frame_annot)

        for seq_name in seq_annot.keys():
            seq_len = len(seq_annot[seq_name])

            step = self.sample_len if self.sample_len > 0 else seq_len
            counter = 0

            for ref_idx in range(0, seq_len, step):
                sample = seq_annot[seq_name][ref_idx : ref_idx + step]
                self.sample_list.append(sample)
                counter += 1
                if only_first_n_samples > 0 and counter >= only_first_n_samples:
                    break

    def __len__(self):
        return len(self.sample_list)

    def crop(self, rgbs, trajs):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        H_new = H
        W_new = W

        # simple random crop
        y0 = 0 if self.crop_size[0] >= H_new else (H_new - self.crop_size[0]) // 2
        x0 = 0 if self.crop_size[1] >= W_new else (W_new - self.crop_size[1]) // 2
        rgbs = [rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for rgb in rgbs]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return rgbs, trajs

    def __getitem__(self, index):
        sample = self.sample_list[index]
        T = len(sample)
        rgbs, visibilities, traj_2d = [], [], []

        H, W = sample[0].image.size
        image_size = (H, W)

        for i in range(T):
            traj_path = os.path.join(self.root, self.split, sample[i].trajectories["path"])
            traj = torch.load(traj_path)

            visibilities.append(traj["verts_inds_vis"].numpy())

            rgbs.append(traj["img"].numpy())
            traj_2d.append(traj["traj_2d"].numpy()[..., :2])

        traj_2d = np.stack(traj_2d)
        visibility = np.stack(visibilities)
        T, N, D = traj_2d.shape
        # subsample trajectories for augmentations
        visible_inds_sampled = torch.randperm(N)[: self.traj_per_sample]

        traj_2d = traj_2d[:, visible_inds_sampled]
        visibility = visibility[:, visible_inds_sampled]

        if self.crop_size is not None:
            rgbs, traj_2d = self.crop(rgbs, traj_2d)
            H, W, _ = rgbs[0].shape
            image_size = self.crop_size

        visibility[traj_2d[:, :, 0] > image_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > image_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        # filter out points that're visible for less than 10 frames
        visible_inds_resampled = visibility.sum(0) > 10
        traj_2d = torch.from_numpy(traj_2d[:, visible_inds_resampled])
        visibility = torch.from_numpy(visibility[:, visible_inds_resampled])

        rgbs = np.stack(rgbs, 0)
        video = torch.from_numpy(rgbs).reshape(T, H, W, 3).permute(0, 3, 1, 2).float()
        return CoTrackerData(
            video=video,
            trajectory=traj_2d,
            visibility=visibility,
            valid=torch.ones(T, N),
            seq_name=sample[0].sequence_name,
        )
