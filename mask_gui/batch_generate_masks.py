#!/usr/bin/env python3
"""
Batch-generate instance masks for every not_reviewed/demo_*** folder.

For each not_reviewed/demo_XXX/demo_XXX_instance.mp4 (the flat-color
instance-segmentation render -- same kind of file the Instance Mask Picker
GUI works on), this builds a mask using the single selected color
(167, 18, 135) in RGB, with the same per-frame color-threshold + cleanup
logic the GUI uses (tolerance 30, morphological open/close cleanup), and
saves the result as:

    not_reviewed/demo_XXX/demo_XXX_instance_mask.npy

-- a boolean array of shape (num_frames, H, W), one frame per index.

Run it directly (requires opencv-python and numpy). With no arguments, it
looks for not_reviewed/ one level up from this script, next to mask_gui/
itself:

    python3 mask_gui/batch_generate_masks.py

Point it at a different folder of demo_*** subfolders with --dir (absolute,
or relative to your current directory):

    python3 mask_gui/batch_generate_masks.py --dir /path/to/some_other_batch
    python3 mask_gui/batch_generate_masks.py --dir ../another_not_reviewed

Or, if your environment doesn't have opencv/numpy installed, run it inside
the same Docker image the GUI uses (it already has both), from the
mask_gui/ directory where docker-compose.yml mounts the project root (one
level up) at /data -- paths after --dir are then container paths under /data:

    cd mask_gui && docker compose exec mask-gui python /data/mask_gui/batch_generate_masks.py --dir /data/not_reviewed
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_NOT_REVIEWED = Path(__file__).resolve().parent.parent / "not_reviewed"  # project root, one level up from mask_gui/

SELECTED_COLORS_RGB = [(167, 18, 135)]  # same color you'd click in the GUI
TOLERANCE = 30.0
USE_CLEANUP = True


# ---------------------------------------------------------------------------
# Same mask-building logic as mask_gui/app.py -- kept in sync on purpose so
# batch output matches what the GUI would produce for the same color/tolerance.
# ---------------------------------------------------------------------------
def color_distance(frame, color_bgr):
    diff = frame.astype(np.int16) - np.array(color_bgr, dtype=np.int16)
    return np.sqrt((diff.astype(np.float32) ** 2).sum(axis=2))


def clean_mask(mask_bool, open_sz=3, close_sz=5):
    m = mask_bool.astype(np.uint8) * 255
    if close_sz > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if open_sz > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_sz, open_sz))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m > 0


def build_mask_stack(video_path, colors_bgr, tolerance, use_cleanup=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    masks = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = np.zeros(frame.shape[:2], dtype=bool)
        for color_bgr in colors_bgr:
            m |= color_distance(frame, color_bgr) < tolerance
        if use_cleanup:
            m = clean_mask(m)
        masks.append(m)
    cap.release()
    if not masks:
        raise RuntimeError(f"no frames read from {video_path}")
    return np.stack(masks, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dir", type=Path, default=DEFAULT_NOT_REVIEWED,
        help=f"folder containing demo_*** subfolders to process (default: {DEFAULT_NOT_REVIEWED})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    not_reviewed = args.dir.resolve()

    if not not_reviewed.is_dir():
        print(f"no such directory: {not_reviewed}", file=sys.stderr)
        sys.exit(1)

    colors_bgr = [(r, g, b)[::-1] for (r, g, b) in SELECTED_COLORS_RGB]

    demo_dirs = sorted(p for p in not_reviewed.iterdir() if p.is_dir() and p.name.startswith("demo_"))
    if not demo_dirs:
        print(f"no demo_* folders found under {not_reviewed}", file=sys.stderr)
        sys.exit(1)

    ok_count, skip_count, fail_count = 0, 0, 0
    for demo_dir in demo_dirs:
        video_path = demo_dir / f"{demo_dir.name}_instance.mp4"
        if not video_path.exists():
            print(f"[skip] {demo_dir.name}: no {video_path.name}")
            skip_count += 1
            continue

        out_path = demo_dir / f"{demo_dir.name}_instance_mask.npy"
        print(f"[{demo_dir.name}] {video_path.name} -> {out_path.name} ...", end=" ", flush=True)
        try:
            stack = build_mask_stack(video_path, colors_bgr, TOLERANCE, USE_CLEANUP)
        except RuntimeError as e:
            print(f"FAILED ({e})")
            fail_count += 1
            continue

        np.save(out_path, stack)
        found = int((stack.sum(axis=(1, 2)) > 0).sum())
        print(f"done. shape={stack.shape}, non-empty frames={found}/{stack.shape[0]}")
        ok_count += 1

    print(f"\n{ok_count} mask(s) written, {skip_count} skipped, {fail_count} failed.")


if __name__ == "__main__":
    main()
