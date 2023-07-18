# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

# from PIL import Image
import imageio
import numpy as np
from cotracker.datasets.utils import CoTrackerData, resize_sample


class FastCaptureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        max_seq_len=50,
        max_num_points=20,
        dataset_resolution=(384, 512),
    ):

        self.data_root = data_root
        self.seq_names = os.listdir(os.path.join(data_root, "renders_local_rm"))
        self.pth_dir = os.path.join(data_root, "zju_tracking")
        self.max_seq_len = max_seq_len
        self.max_num_points = max_num_points
        self.dataset_resolution = dataset_resolution
        print("found %d unique videos in %s" % (len(self.seq_names), self.data_root))

    def __getitem__(self, index):
        seq_name = self.seq_names[index]
        spath = os.path.join(self.data_root, "renders_local_rm", seq_name)
        pthpath = os.path.join(self.pth_dir, seq_name + ".pth")

        rgbs = []
        img_paths = sorted(os.listdir(spath))
        for i, img_path in enumerate(img_paths):
            if i < self.max_seq_len:
                rgbs.append(imageio.imread(os.path.join(spath, img_path)))

        annot_dict = torch.load(pthpath)
        traj_2d = annot_dict["traj_2d"][:, :, : self.max_seq_len]
        visibility = annot_dict["visibility"][:, : self.max_seq_len]

        S = len(rgbs)
        H, W, __ = rgbs[0].shape
        *_, S = traj_2d.shape
        visibile_pts_first_frame_inds = (visibility[:, 0] > 0).nonzero(as_tuple=False)[
            :, 0
        ]
        torch.manual_seed(0)
        point_inds = torch.randperm(len(visibile_pts_first_frame_inds))[
            : self.max_num_points
        ]
        visible_inds_sampled = visibile_pts_first_frame_inds[point_inds]

        rgbs = np.stack(rgbs, 0)
        rgbs = torch.from_numpy(rgbs).reshape(S, H, W, 3).permute(0, 3, 1, 2).float()

        segs = torch.ones(S, 1, H, W).float()
        trajs = traj_2d[visible_inds_sampled].permute(2, 0, 1).float()
        visibles = visibility[visible_inds_sampled].permute(1, 0)

        rgbs, trajs, segs = resize_sample(rgbs, trajs, segs, self.dataset_resolution)

        return CoTrackerData(rgbs, segs, trajs, visibles, seq_name=seq_name)

    def __len__(self):
        return len(self.seq_names)
