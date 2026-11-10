"""Build a (T, H, W) bool mask .npy from a flat-color instance-segmentation video.

Same per-frame logic as mask_gui (imported from mask_gui/batch_generate_masks.py so the two stay
in sync): a pixel is in the mask if its color is within --tolerance (Euclidean RGB distance) of
any selected color, then a morphological close + open cleans it up.

Example:
    python scripts/extract_mask.py --video demo_008/demo_008_instance.mp4 \
        --out demo_008_instance_mask.npy --colors "167,18,135"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for mask_gui
from mask_gui.batch_generate_masks import clean_mask, color_distance  # noqa: E402


def parse_colors(spec):
    """'167,18,135' or '167,18,135; 20,200,40' -> [(167, 18, 135), ...] (RGB)."""
    colors = []
    for part in spec.replace("|", ";").split(";"):
        if part.strip():
            rgb = tuple(int(v) for v in part.replace(" ", "").split(","))
            if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
                raise ValueError(f"bad color {part!r}: expected R,G,B with values 0-255")
            colors.append(rgb)
    if not colors:
        raise ValueError("no colors given")
    return colors


def extract_mask(video, colors_rgb, tolerance=30.0, cleanup=True, open_sz=3, close_sz=5):
    colors_bgr = [c[::-1] for c in colors_rgb]  # OpenCV frames are BGR
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    masks = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = np.zeros(frame.shape[:2], dtype=bool)
        for c in colors_bgr:
            m |= color_distance(frame, c) < tolerance
        if cleanup:
            m = clean_mask(m, open_sz=open_sz, close_sz=close_sz)
        masks.append(m)
    cap.release()
    if not masks:
        raise RuntimeError(f"no frames read from {video}")
    return np.stack(masks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="instance-segmentation video")
    p.add_argument("--out", required=True, help="output .npy")
    p.add_argument("--colors", default="167,18,135", help='RGB, e.g. "167,18,135"; several with ";"')
    p.add_argument("--tolerance", type=float, default=30.0)
    p.add_argument("--no_cleanup", action="store_true")
    p.add_argument("--open", type=int, default=3, help="morphological open kernel (px), 0 = off")
    p.add_argument("--close", type=int, default=5, help="morphological close kernel (px), 0 = off")
    args = p.parse_args()
    stack = extract_mask(args.video, parse_colors(args.colors), args.tolerance,
                         not args.no_cleanup, args.open, args.close)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, stack)
    found = int((stack.sum(axis=(1, 2)) > 0).sum())
    print(f"mask {stack.shape} -> {args.out}; non-empty frames {found}/{len(stack)}")


if __name__ == "__main__":
    main()
