"""Make the video set of a VLM question loop: a folder with links to the chosen demo folders, and labels.csv.

    python scripts/vlm_loop/make_set.py --demos_dir DIR --demos 0-49,54,56,58 --exclude 45,47,48 \
        --verdicts videos.csv --video_col demo --verdict_col verdict --good approved,weak --bad rejected \
        --out SET_DIR

--demos_dir holds one folder per demo (demo_000, demo_001, ...), each with the generated video and its
instance video (see --need). --verdicts is a CSV with one row per demo: --video_col gives the demo (a number,
or the folder name) and --verdict_col its review verdict; --good and --bad list which verdicts count as good
and bad. A chosen demo without a folder, without the needed files, or with another verdict stops the script.

--first_good N --first_bad M pick, in the order of --demos, the first N good and the first M bad demos; demos
with another verdict (for example not reviewed) are skipped. When fewer are there, all of them are used and
the script says so. --exclude_set SET_DIR leaves out the demos of another set (for example the train set when
making the test set).

    train: --demos 0-99 --first_good 25 --first_bad 25 --out train_set
    test:  --demos 0-99 --exclude_set train_set --first_good 25 --first_bad 25 --out test_set
           (without --first_good/--first_bad: all other demos)

Writes SET_DIR/demo_XXX (links to the demo folders) and SET_DIR/labels.csv (video,label). An existing SET_DIR
is only reused when it has exactly the same demos and labels.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def parse_demos(spec: str) -> list[int]:
    """'0-3,7' -> [0, 1, 2, 3, 7]."""
    out = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def demo_name(value: str, name_format: str) -> str:
    """A demo number ('45') or a folder name ('demo_045') -> the folder name."""
    value = value.strip()
    return name_format.format(int(value)) if value.isdigit() else value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos_dir", type=Path, required=True)
    p.add_argument("--demos", required=True, help="demo numbers, e.g. 0-49,54")
    p.add_argument("--exclude", default="", help="demo numbers to leave out, e.g. 45,47,48")
    p.add_argument("--verdicts", type=Path, required=True, help="CSV with one row per demo")
    p.add_argument("--video_col", default="demo")
    p.add_argument("--verdict_col", default="verdict")
    p.add_argument("--good", required=True, help="verdicts that count as good, comma separated")
    p.add_argument("--bad", required=True, help="verdicts that count as bad, comma separated")
    p.add_argument("--name_format", default="demo_{:03d}", help="folder name of a demo number")
    p.add_argument("--need", default="_01.mp4,_instance_all.mp4",
                   help="files every demo folder must have, as suffixes after the folder name")
    p.add_argument("--first_good", type=int, help="take the first N good demos (with --first_bad)")
    p.add_argument("--first_bad", type=int, help="take the first M bad demos (with --first_good)")
    p.add_argument("--exclude_set", type=Path, help="a set folder whose demos are left out")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    pick = args.first_good is not None or args.first_bad is not None
    if pick and (args.first_good is None or args.first_bad is None):
        p.error("--first_good and --first_bad go together")

    good, bad = set(args.good.split(",")), set(args.bad.split(","))
    if good & bad:
        sys.exit(f"verdicts in both --good and --bad: {', '.join(sorted(good & bad))}")
    skip = {args.name_format.format(d) for d in parse_demos(args.exclude)}
    if args.exclude_set:
        with open(args.exclude_set / "labels.csv", newline="") as f:
            skip |= {r["video"] for r in csv.DictReader(f)}
    names = [args.name_format.format(d) for d in parse_demos(args.demos)]
    if len(set(names)) != len(names):
        sys.exit("a demo is listed twice in --demos")
    names = [n for n in names if n not in skip]

    verdict = {}
    with open(args.verdicts, newline="") as f:
        for row in csv.DictReader(f):
            name = demo_name(row[args.video_col], args.name_format)
            if name in verdict:
                sys.exit(f"{args.verdicts}: more than one row for {name}")
            verdict[name] = row[args.verdict_col].strip()

    if pick:  # the first N good and M bad in the order of --demos; other verdicts are skipped
        want = {"good": args.first_good, "bad": args.first_bad}
        chosen = []
        for name in names:
            v = verdict.get(name)
            label = "good" if v in good else "bad" if v in bad else None
            if label and want[label] > 0:
                chosen.append(name)
                want[label] -= 1
        for label, left in want.items():
            if left > 0:
                n = args.first_good if label == "good" else args.first_bad
                print(f"only {n - left} {label} demos left, not {n}: all of them are used")
        names = chosen

    rows, errors = [], []
    for name in names:
        d = args.demos_dir / name
        missing = [s for s in args.need.split(",") if s and not (d / f"{name}{s}").is_file()]
        v = verdict.get(name)
        if not d.is_dir():
            errors.append(f"{name}: no folder {d}")
        elif missing:
            errors.append(f"{name}: missing {', '.join(name + s for s in missing)}")
        elif v not in good | bad:
            errors.append(f"{name}: verdict {v!r} is neither good nor bad")
        else:
            rows.append((name, "good" if v in good else "bad"))
    if errors:
        sys.exit("\n".join(errors))

    labels = args.out / "labels.csv"
    if args.out.exists():
        old = sorted(p.name for p in args.out.iterdir() if p.name.startswith("demo_"))
        with open(labels, newline="") as f:
            old_rows = [(r["video"], r["label"]) for r in csv.DictReader(f)] if labels.is_file() else []
        if old != sorted(n for n, _ in rows) or sorted(old_rows) != sorted(rows):
            sys.exit(f"{args.out} exists with other demos or labels; use a new folder")
        print(f"{args.out}: already made, same demos and labels")
        return
    args.out.mkdir(parents=True)
    for name, _ in rows:
        os.symlink((args.demos_dir / name).resolve(), args.out / name)
    with open(labels, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["video", "label"])
        w.writerows(sorted(rows))
    n_good = sum(label == "good" for _, label in rows)
    print(f"{args.out}: {len(rows)} demos, {n_good} good, {len(rows) - n_good} bad")


if __name__ == "__main__":
    main()
