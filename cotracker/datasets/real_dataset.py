# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import json
import cv2
import math
import imageio
import numpy as np

from cotracker.datasets.utils import CoTrackerData
from torchvision.transforms import ColorJitter, GaussianBlur
from PIL import Image
from cotracker.models.core.model_utils import smart_cat
from torchvision.io import read_video
import torchvision
from cotracker.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
import torchvision.transforms.functional as F


class RealDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        random_frame_rate=False,
        random_seq_len=False,
        data_splits=[0],
        random_resize=False,
        limit_samples=10000,
    ):
        super(RealDataset, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)
        raise ValueError(f"This dataset wasn't released. You should collect your own dataset of real videos before training with this dataset class.")

        stopwords = set(
            [
                "river",
                "water",
                "shore",
                "lake",
                "sea",
                "ocean",
                "silhouette",
                "matte",
                "online",
                "virtual",
                "meditation",
                "artwork",
                "drawing",
                "animation",
                "abstract",
                "background",
                "concept",
                "cartoon",
                "symbolic",
                "painting",
                "sketch",
                "fireworks",
                "fire",
                "sky",
                "darkness",
                "timelapse",
                "time-lapse",
                "cgi",
                "computer",
                "computer-generated",
                "drawing",
                "draw",
                "cgi",
                "animate",
                "cartoon",
                "static",
                "abstract",
                "abstraction",
                "3d",
                "fandom",
                "fantasy",
                "graphics",
                "cell",
                "holographic",
                "generated",
                "generation" "telephoto",
                "animated",
                "disko",
                "generate" "2d",
                "3d",
                "geometric",
                "geometry",
                "render",
                "rendering",
                "timelapse",
                "slomo",
                "slo",
                "wallpaper",
                "pattern",
                "tile",
                "generated",
                "chroma",
                "www",
                "http",
                "cannabis",
                "loop",
                "cycle",
                "alpha",
                "abstract",
                "concept",
                "digital",
                "graphic",
                "skies",
                "fountain",
                "train",
                "rapid",
                "fast",
                "quick",
                "vfx",
                "effect",
            ]
        )

        def no_stopwords_in_key(key, stopwords):
            for s in stopwords:
                if s in key.split(","):
                    return False
            return True

        filelist_all = []

        for part in data_splits:
            filelist = np.load('YOUR FILELIST')
            captions = np.load('YOUR CAPTIONS')
            keywords = np.load('YOUR KEYWORDS')

            filtered_seqs_motion = [
                i
                for i, key in enumerate(keywords)
                if "motion" in key.split(",")
                and (
                    "man" in key.split(",")
                    or "woman" in key.split(",")
                    or "animal" in key.split(",")
                    or "child" in key.split(",")
                )
                and no_stopwords_in_key(key, stopwords)
            ]
            print("filtered_seqs_motion", len(filtered_seqs_motion))
            filtered_seqs = filtered_seqs_motion

            print(f"filtered_seqs {part}", len(filtered_seqs))
            filelist_all = filelist_all + filelist[filtered_seqs].tolist()

            if len(filelist_all) > limit_samples:
                break

        self.filelist = filelist_all[:limit_samples]
        print(f"found {len(self.filelist)} unique videos")
        self.traj_per_sample = traj_per_sample
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.random_frame_rate = random_frame_rate
        self.random_resize = random_resize
        self.random_seq_len = random_seq_len

    def crop(self, rgbs):
        S = len(rgbs)

        H, W = rgbs.shape[2:]

        H_new = H
        W_new = W

        # simple random crop
        y0 = (
            0
            if self.crop_size[0] >= H_new
            else np.random.randint(0, H_new - self.crop_size[0])
        )
        x0 = (
            0
            if self.crop_size[1] >= W_new
            else np.random.randint(0, W_new - self.crop_size[1])
        )
        rgbs = [
            rgb[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            for rgb in rgbs
        ]

        return torch.stack(rgbs)

    def __getitem__(self, index):
        gotit = False

        sample, gotit = self.getitem_helper(index)
        if not gotit:
            print("warning: sampling failed")
            # fake sample, so we can still collate
            sample = CoTrackerData(
                video=torch.zeros(
                    (self.seq_len, 3, self.crop_size[0], self.crop_size[1])
                ),
                trajectory=torch.ones(1, 1, 1, 2),
                visibility=torch.ones(1, 1, 1),
                valid=torch.ones(1, 1, 1),
            )

        return sample, gotit

    def sample_h_w(self):
        area = np.random.uniform(0.6, 1)
        a1 = np.random.uniform(area, 1)
        a2 = np.random.uniform(area, 1)
        h = (a1 + a2) / 2.0
        w = area / h
        return h, w

    def getitem_helper(self, index):
        gotit = True
        video_path = self.filelist[index]

        rgbs, _, _ = read_video(str(video_path), output_format="TCHW", pts_unit="sec")
        if rgbs.numel() == 0:
            return None, False
        seq_name = video_path
        frame_rate = 1

        if self.random_seq_len:
            seq_len = np.random.randint(int(self.seq_len / 2), self.seq_len)
        else:
            seq_len = self.seq_len

        while len(rgbs) < seq_len:
            rgbs = torch.cat([rgbs, rgbs.flip(0)])
        if seq_len < 8:
            print("seq_len < 8, return NONE")
            return None, False
        if self.random_frame_rate:
            max_frame_rate = min(4, int((len(rgbs) / seq_len)))
            if max_frame_rate > 1:
                frame_rate = np.random.randint(1, max_frame_rate)

        if seq_len * frame_rate < len(rgbs):
            start_ind = np.random.choice(len(rgbs) - (seq_len * frame_rate), 1)[0]
        else:
            start_ind = 0
        rgbs = rgbs[start_ind : start_ind + seq_len * frame_rate : frame_rate]

        assert seq_len <= len(rgbs)

        if self.random_resize and np.random.rand() < 0.5:
            video = []
            rgbs = rgbs.permute(0, 2, 3, 1).numpy()

            for i in range(len(rgbs)):
                rgb = cv2.resize(
                    rgbs[i],
                    (self.crop_size[1], self.crop_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                video.append(rgb)
            video = torch.tensor(np.stack(video)).permute(0, 3, 1, 2)

        else:
            video = self.crop(rgbs)

        sample = CoTrackerData(
            video=video,
            trajectory=torch.ones(seq_len, self.traj_per_sample, 2),
            visibility=torch.ones(seq_len, self.traj_per_sample),
            valid=torch.ones(seq_len, self.traj_per_sample),
            seq_name=seq_name,
        )

        return sample, gotit

    def __len__(self):
        return len(self.filelist)
