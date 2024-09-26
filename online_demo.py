# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
from PIL import Image
import numpy as np

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def _process_step(model, frames, is_first_step, grid_size, grid_query_frame, segm_mask):
    video_chunk = (
        torch.tensor(np.stack(frames[-model.step * 2 :]), device="cuda")
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)
    return model(
        video_chunk,
        is_first_step=is_first_step,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
        segm_mask=segm_mask
    )

def _process_forward(model, grid_size, segm_mask, frames, grid_query_frame=0):

        """
        frames -> F X C X H X W
        """

        window_frames = []
        is_first_step = True

        for i, frame in enumerate(
            frames
        ):
            if i % model.step == 0 and i != 0:
                pred_tracks, pred_visibility = _process_step(
                    model,
                    window_frames,
                    is_first_step,
                    grid_size=grid_size,
                    grid_query_frame=grid_query_frame,
                    segm_mask=segm_mask
                )
                is_first_step = False
            window_frames.append(frame)
        # Processing the final video frames in case video length is not a multiple of model.step
        pred_tracks, pred_visibility = _process_step(
            model,
            window_frames[-(i % model.step) - model.step - 1 :],
            is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            segm_mask=segm_mask
        )

        return pred_tracks, pred_visibility


def get_frames(video_path):

    frames = []
    for i, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
        frames.append(frame)

    return frames

def predict(window_frames, model, grid_size, segm_mask=None, grid_query_frame=0, backward_tracking=False):
    
    pred_tracks, pred_visibility = _process_forward(model, grid_size, segm_mask, window_frames, grid_query_frame)

    if backward_tracking:   # flip frames and treat as forward
        frames_backward = window_frames[:grid_query_frame+1][::-1]
        pred_tracks_backward, pred_visibility_backward = _process_forward(model, grid_size, segm_mask, frames_backward, 0)

        pred_tracks_backward = torch.flip(pred_tracks_backward[:,1:,:,:], dims=[1])
        pred_visibility_backward = torch.flip(pred_visibility_backward[:,1:,:], dims=[1])

        pred_tracks[:, 0:grid_query_frame] = pred_tracks_backward
        pred_visibility[:, 0:grid_query_frame] = pred_visibility_backward

    return pred_tracks, pred_visibility
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cotracker2.pth",
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=20,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="./assets/apple_mask.png",
        help="Path to segmentation mask for grid initialization",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Perform backward"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = get_frames(args.video_path)

    segm_mask = None
    if args.mask_path is not None:
        segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
        segm_mask = torch.from_numpy(segm_mask)[None, None]

    if args.backward_tracking and args.grid_query_frame < model.step:
        print("Backward tracking is not possible with grid_query_frame < model.step")
        print("Backward tracking deactivated")
        args.backward_tracking = False

    pred_tracks, pred_visibility = predict(window_frames, model, args.grid_size, segm_mask, args.grid_query_frame, args.backward_tracking)

    print("Tracks are computed")

    # save a video with predicted tracks
    seq_name = os.path.splitext(args.video_path.split("/")[-1])[0]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(video, pred_tracks, pred_visibility, query_frame=0 if args.backward_tracking else args.grid_query_frame, filename=seq_name)
