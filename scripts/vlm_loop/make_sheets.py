"""Frame sheets for the question writer of a VLM question loop.

    python scripts/vlm_loop/make_sheets.py --share SHARE_DIR --sets SETS_CSV --set SET OUT_DIR \
        --towel_color "R,G,B; R,G,B" [--width 640]

For every video of the set (the rows of --set in SETS_CSV, made by make_set.py; the demo folders are read straight
from the share folder) two JPEG sheets with 11 rows, the moments 0.0, 0.1, ..., 1.0 of the generated video
(<demo>_01.mp4); the moment is written in the top left corner of each row:
    OUT_DIR/frames11/<demo>.jpg  the full frame (third-person view left, wrist view right), scaled to --width
    OUT_DIR/crops11/<demo>.jpg   the object close-ups: left the third-person half, right the wrist half, made
                                 with towel_box of vlm_check.py on the instance video (<demo>_instance_all.mp4),
                                 the same close-up the checker gets; a half without the object is shown whole
A sheet that is already there is kept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]  # vlm_check, extract_mask (needs mask_gui)
import vlm_check  # noqa: E402
from extract_mask import parse_colors  # noqa: E402
from make_set import set_folders  # noqa: E402

MOMENTS = [k / 10 for k in range(11)]  # the prompt names the folders frames11 and crops11
CELL = 320  # side of one close-up on the sheet, px


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(img))
    ImageDraw.Draw(im).text((4, 4), text, fill=(255, 255, 0))
    return np.asarray(im)


def sheets(demo: Path, out: Path, colors, width: int, video_suffix: str, instance_suffix: str):
    video, inst = demo / f"{demo.name}{video_suffix}.mp4", demo / f"{demo.name}{instance_suffix}.mp4"
    n = vlm_check.get_frame_count(video)
    idx = [round(a * (n - 1)) for a in MOMENTS]
    frames = vlm_check.read_frames(video, set(idx))
    insts = {}
    if inst.is_file():
        ni = vlm_check.get_frame_count(inst)
        got = vlm_check.read_frames(inst, {min(i, ni - 1) for i in idx})
        insts = {i: got[min(i, ni - 1)] for i in idx}
    full_rows, crop_rows = [], []
    for a, i in zip(MOMENTS, idx):
        img = frames[i]
        h, w = img.shape[:2]
        row = Image.fromarray(img)
        if width and w != width:
            row = row.resize((width, round(h * width / w)), Image.BICUBIC)
        full_rows.append(label(np.asarray(row), f"{a:.1f}"))
        half, cells = w // 2, []
        for x in (0, half):
            part = Image.fromarray(img[:, x:x + half])
            box = vlm_check.towel_box(insts[i][:, x:x + half], colors) if i in insts else None
            crop = part.crop(box) if box else part
            cells.append(np.asarray(crop.resize((CELL, CELL), Image.BICUBIC)))
        crop_rows.append(label(np.concatenate(cells, axis=1), f"{a:.1f}"))
    Image.fromarray(np.concatenate(full_rows, axis=0)).save(out / "frames11" / f"{demo.name}.jpg", quality=90)
    Image.fromarray(np.concatenate(crop_rows, axis=0)).save(out / "crops11" / f"{demo.name}.jpg", quality=90)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--share", type=Path, required=True, help="the share folder with the videos (share.py)")
    p.add_argument("--sets", type=Path, required=True, help="the sets CSV (video, label, set; make_set.py)")
    p.add_argument("--set", required=True, help="the set, e.g. train")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--towel_color", required=True, help="instance colors of the object, 'R,G,B; R,G,B'")
    p.add_argument("--width", type=int, default=640, help="width of a full-frame row, px (0 = as recorded)")
    p.add_argument("--video_suffix", default="_01")
    p.add_argument("--instance_suffix", default="_instance_all")
    args = p.parse_args()
    colors = parse_colors(args.towel_color)
    for sub in ("frames11", "crops11"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)
    demos = [d for _, d in sorted(set_folders(args.share, args.sets, args.set).items())]
    for d in demos:
        if all((args.out_dir / sub / f"{d.name}.jpg").is_file() for sub in ("frames11", "crops11")):
            continue
        sheets(d, args.out_dir, colors, args.width, args.video_suffix, args.instance_suffix)
        print(d.name, flush=True)
    print(f"{len(demos)} demos: {args.out_dir / 'frames11'}, {args.out_dir / 'crops11'}")


if __name__ == "__main__":
    main()
