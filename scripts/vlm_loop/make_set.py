"""Make a video set of a VLM question loop: the rows of one set in a sets CSV (video, label, set), from a share folder.

    python scripts/vlm_loop/make_set.py --share SHARE_DIR --demos 0-99 --first_good 50 --first_bad 50 \
        --sets SETS_CSV --set train
    python scripts/vlm_loop/make_set.py --share SHARE_DIR --demos 0-99 --first_good 50 --first_bad 50 \
        --sets SETS_CSV --set test
    python scripts/vlm_loop/make_set.py --share SHARE_DIR --demos 0-749 --sets SETS_CSV --set all

SHARE_DIR is a share folder (made by share.py): the videos are in videos/<O|weak|X|not_reviewed>/demo_NNN/, and
the review of the current video of every demo is in lists/ (videos.csv and review.csv). --good and --bad list
which verdicts count as good and bad. No folder or link is made: make_sheets.py and run_batch.py read the
videos of a set straight from the share folder.

--first_good N --first_bad M pick, in the order of --demos, the first N good and the first M bad demos; demos
with another verdict (for example not reviewed) are skipped. When fewer are there, all of them are used and
the script says so. Without them, every demo of --demos is used, and each must be good or bad; with
--reviewed_only the demos that are neither (for example not reviewed) are left out, and the script says how many.
Demos that are in another set of SETS_CSV are left out, so a test set made after the train set has other videos.
A chosen demo without its folder in the share folder or without the needed files (--need) stops the script.

Adds the rows of --set to SETS_CSV (made when missing). A set that is already in SETS_CSV is only kept when it
has exactly the same videos and labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from score import share_verdicts

NAME = "demo_{:03d}"  # the folder name of a demo number in a share folder


def parse_demos(spec: str) -> list[int]:
    """'0-3,7' -> [0, 1, 2, 3, 7]."""
    out = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def read_sets(sets_csv: Path) -> list[dict]:
    """The rows (video, label, set) of a sets CSV; none when the file is not there."""
    if not sets_csv.is_file():
        return []
    with open(sets_csv, newline="") as f:
        return list(csv.DictReader(f))


def share_demos(share: Path) -> dict:
    """{demo folder name: the folder} of the videos of a share folder (videos/<O|weak|X|not_reviewed>/demo_NNN)."""
    found = {}
    for d in sorted((share / "videos").glob("*/demo_*")):
        if d.is_dir():
            if d.name in found:
                sys.exit(f"{share}: {d.name} is in two folders, {found[d.name].parent.name} and {d.parent.name}")
            found[d.name] = d
    return found


def set_folders(share: Path, sets_csv: Path, set_name: str) -> dict:
    """{demo folder name: its folder in the share folder} of the videos of one set of a sets CSV."""
    names = [r["video"] for r in read_sets(sets_csv) if r["set"] == set_name]
    if not names:
        sys.exit(f"no videos of set {set_name} in {sets_csv}")
    found = share_demos(share)
    missing = [n for n in names if n not in found]
    if missing:
        sys.exit(f"{len(missing)} videos of set {set_name} are not in {share / 'videos'}: {', '.join(missing[:5])}")
    return {n: found[n] for n in names}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--share", type=Path, required=True, help="the share folder of the dataset (share.py)")
    p.add_argument("--demos", required=True, help="demo numbers, e.g. 0-49,54")
    p.add_argument("--exclude", default="", help="demo numbers to leave out, e.g. 45,47,48")
    p.add_argument("--good", default="approved,weak", help="verdicts that count as good, comma separated")
    p.add_argument("--bad", default="rejected", help="verdicts that count as bad, comma separated")
    p.add_argument("--need", default="_01.mp4,_instance_all.mp4",
                   help="files every demo folder must have, as suffixes after the folder name")
    p.add_argument("--first_good", type=int, help="take the first N good demos (with --first_bad)")
    p.add_argument("--first_bad", type=int, help="take the first M bad demos (with --first_good)")
    p.add_argument("--reviewed_only", action="store_true",
                   help="without --first_good/--first_bad: leave out the demos that are neither good nor bad")
    p.add_argument("--sets", type=Path, required=True, help="the sets CSV (video, label, set), made when missing")
    p.add_argument("--set", required=True, help="the name of the set, e.g. train, test or all")
    args = p.parse_args()
    pick = args.first_good is not None or args.first_bad is not None
    if pick and (args.first_good is None or args.first_bad is None):
        p.error("--first_good and --first_bad go together")

    good, bad = set(args.good.split(",")), set(args.bad.split(","))
    if good & bad:
        sys.exit(f"verdicts in both --good and --bad: {', '.join(sorted(good & bad))}")
    old = read_sets(args.sets)
    skip = {NAME.format(d) for d in parse_demos(args.exclude)} | {r["video"] for r in old if r["set"] != args.set}
    names = [NAME.format(d) for d in parse_demos(args.demos)]
    if len(set(names)) != len(names):
        sys.exit("a demo is listed twice in --demos")
    names = [n for n in names if n not in skip]
    verdict = {NAME.format(n): v for n, v in share_verdicts(args.share / "lists").items()}

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
    elif args.reviewed_only:
        left_out = [n for n in names if verdict.get(n) not in good | bad]
        if left_out:
            print(f"{len(left_out)} demos are neither good nor bad (for example not reviewed): left out")
        names = [n for n in names if n not in left_out]

    found = share_demos(args.share)
    rows, errors = [], []
    for name in names:
        d = found.get(name)
        missing = [s for s in args.need.split(",") if s and d and not (d / f"{name}{s}").is_file()]
        v = verdict.get(name)
        if d is None:
            errors.append(f"{name}: no folder in {args.share / 'videos'}")
        elif missing:
            errors.append(f"{name}: missing {', '.join(name + s for s in missing)}")
        elif v not in good | bad:
            errors.append(f"{name}: verdict {v!r} is neither good nor bad")
        else:
            rows.append((name, "good" if v in good else "bad"))
    if errors:
        sys.exit("\n".join(errors))

    mine = sorted((r["video"], r["label"]) for r in old if r["set"] == args.set)
    if mine:
        if mine != sorted(rows):
            sys.exit(f"set {args.set} is in {args.sets} with other videos or labels; use another set name or file")
        print(f"{args.sets}: set {args.set} already made, same videos and labels")
        return
    args.sets.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sets, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["video", "label", "set"])
        w.writerows((r["video"], r["label"], r["set"]) for r in old)
        w.writerows((name, label, args.set) for name, label in sorted(rows))
    n_good = sum(label == "good" for _, label in rows)
    print(f"{args.sets}: set {args.set}, {len(rows)} demos, {n_good} good, {len(rows) - n_good} bad")


if __name__ == "__main__":
    main()
