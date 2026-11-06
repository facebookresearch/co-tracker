"""Score one VLM batch against the labels of the set.

    python scripts/vlm_loop/score.py BATCH_OUT LABELS_CSV OUT_DIR [--calibrate [max_right]] [--video_suffix _01]
    python scripts/vlm_loop/score.py --pyes PYES_CSV --questions QUESTIONS_JSON --sets SETS_CSV --set SET OUT_DIR \
        [--calibrate [max_right]]

BATCH_OUT is the GUI output folder of the batch (outputs/gui/<batch name>), with <demo>/<demo>_01_vlm.json.
A video is marked bad when any question is Yes (P(yes) > the question's threshold).

--pyes scores saved P(yes) values instead of a batch (Qwen does not run), for example the files in
scripts/vlm_loop/results/vlm_v1.0/. PYES_CSV has one row per video: video, then <question id>_p_yes for every
question. The thresholds are the ones in QUESTIONS_JSON. SETS_CSV has the columns video, label (good or bad) and
set; only the videos of --set (for example train, test or other) are scored.

--calibrate sets every threshold from the P(yes) values of this batch instead of using the thresholds the
questions were run with: halfway between the highest P(yes) of a good video and the next higher P(yes) of a bad
video, so the question gives no good video a Yes and keeps the largest distance to the videos on both sides.
A question where no bad video is above every good video gets a threshold halfway between the highest good
P(yes) and 1 (it then catches nothing). The batch does not have to run again: only the comparison changes.

--calibrate max_right then lowers these thresholds where this gives more right videos on this batch. A bad
video without a Yes and a good video with a Yes count the same: one wrong video. For each question in turn, the
threshold moves to the highest value with the fewest wrong videos, but only when that is fewer than now. The
other thresholds stay as they are. This repeats until no threshold changes. The values tried are 1 and the
midpoints between neighboring P(yes) values of the question. With this rule, good videos can get a Yes.

Writes:
    OUT_DIR/score.json        correct, bad_caught, good_passed, bad_passed (bad videos marked good),
                              good_flagged, the Yes list of every question, the margin (for every question
                              the smallest distance between its threshold and any video's P(yes), and the
                              smallest of these), and the questions with the thresholds used
    OUT_DIR/pyes.csv          one row per video: label, verdict and the P(yes) of every question
  with --calibrate also:
    OUT_DIR/score_as_run.json the same with the thresholds the questions were run with
    OUT_DIR/questions_calibrated.json  the questions with the set thresholds
Prints one summary line.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(batch_out: Path, labels_csv: Path, video_suffix: str) -> tuple[dict, list[dict], dict]:
    """-> (labels {video: good|bad}, questions as run, P(yes) {video: {question id: p}})."""
    with open(labels_csv, newline="") as f:
        labels = {r["video"]: r["label"] for r in csv.DictReader(f)}
    pyes, questions = {}, None
    for v in sorted(labels):
        res = json.load(open(batch_out / v / f"{v}{video_suffix}_vlm.json"))
        pyes[v] = {q: a["p_yes"] for q, a in res["answers"].items()}
        questions = questions or res["questions"]
    return labels, questions, pyes


def load_saved(pyes_csv: Path, questions_json: Path, sets_csv: Path, set_name: str) -> tuple[dict, list[dict], dict]:
    """-> (labels {video: good|bad} of one set, questions with their thresholds, P(yes) {video: {question id: p}})."""
    with open(sets_csv, newline="") as f:
        labels = {r["video"]: r["label"] for r in csv.DictReader(f) if r["set"] == set_name}
    if not labels:
        raise SystemExit(f"no videos of set {set_name} in {sets_csv}")
    questions = json.load(open(questions_json))
    with open(pyes_csv, newline="") as f:
        saved = {r["video"]: r for r in csv.DictReader(f)}
    missing = sorted(set(labels) - set(saved))
    if missing:
        raise SystemExit(f"no P(yes) in {pyes_csv} for {len(missing)} videos: {', '.join(missing[:5])}")
    pyes = {v: {q["id"]: float(saved[v][f"{q['id']}_p_yes"]) for q in questions} for v in labels}
    return labels, questions, pyes


def calibrate(questions: list[dict], pyes: dict, labels: dict, max_right: bool = False) -> list[dict]:
    """The questions with thresholds set from the P(yes) values (see --calibrate and --calibrate max_right)."""
    out = []
    for q in questions:
        good_max = max(pyes[v][q["id"]] for v in labels if labels[v] == "good")
        above = [pyes[v][q["id"]] for v in labels if labels[v] == "bad" and pyes[v][q["id"]] > good_max]
        upper = min(above) if above else 1.0
        out.append(dict(q, threshold=(good_max + upper) / 2))
    return lower_for_max_right(out, pyes, labels) if max_right else out


def wrong_count(thr: dict, pyes: dict, labels: dict) -> int:
    """Bad videos without a Yes plus good videos with a Yes, with the thresholds thr {question id: threshold}."""
    return sum(any(pyes[v][q] > t for q, t in thr.items()) != (labels[v] == "bad") for v in labels)


def lower_for_max_right(questions: list[dict], pyes: dict, labels: dict) -> list[dict]:
    """The max_right rule (see --calibrate max_right), starting from the thresholds of the questions."""
    thr = {q["id"]: q["threshold"] for q in questions}
    tried = {}
    for q in thr:
        values = sorted({pyes[v][q] for v in labels})
        tried[q] = sorted({(a + b) / 2 for a, b in zip(values, values[1:])} | {1.0}, reverse=True)
    best = wrong_count(thr, pyes, labels)
    changed = True
    while changed:
        changed = False
        for q in thr:
            for t in tried[q]:  # high to low: of equal counts, the highest threshold is kept
                n = wrong_count({**thr, q: t}, pyes, labels)
                if n < best:
                    best, thr[q], changed = n, t, True
    return [dict(q, threshold=thr[q["id"]]) for q in questions]


def score_with(questions: list[dict], pyes: dict, labels: dict) -> tuple[dict, list[dict]]:
    qids = [q["id"] for q in questions]
    thr = {q["id"]: q["threshold"] for q in questions}
    yes = {v: {q: pyes[v][q] > thr[q] for q in qids} for v in labels}
    verdict = {v: "bad" if any(yes[v].values()) else "good" for v in labels}
    bad = [v for v in labels if labels[v] == "bad"]
    good = [v for v in labels if labels[v] == "good"]
    margins = {q: round(min(abs(pyes[v][q] - thr[q]) for v in labels), 4) for q in qids}
    s = {
        "videos": len(labels),
        "correct": sum(verdict[v] == labels[v] for v in labels),
        "bad_caught": sum(verdict[v] == "bad" for v in bad), "bad_total": len(bad),
        "good_passed": sum(verdict[v] == "good" for v in good), "good_total": len(good),
        "bad_passed": [v for v in bad if verdict[v] == "good"],
        "good_flagged": [v for v in good if verdict[v] == "bad"],
        "yes_per_question": {q: {"bad": [v for v in bad if yes[v][q]], "good": [v for v in good if yes[v][q]]}
                             for q in qids},
        "margin_per_question": margins,
        "margin": min(margins.values()),
        "questions": questions,
    }
    rows = [dict(video=v, label=labels[v], verdict=verdict[v], **{f"{q}_p_yes": pyes[v][q] for q in qids})
            for v in sorted(labels)]
    return s, rows


def score(batch_out: Path, labels_csv: Path, video_suffix: str = "_01") -> tuple[dict, list[dict]]:
    """The batch with the thresholds it was run with."""
    labels, questions, pyes = load(batch_out, labels_csv, video_suffix)
    return score_with(questions, pyes, labels)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_out", type=Path, nargs="?")
    p.add_argument("labels", type=Path, nargs="?")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--calibrate", nargs="?", const=True, choices=["max_right"],
                   help="set the thresholds from this batch's P(yes); max_right: the max_right rule (see above)")
    p.add_argument("--video_suffix", default="_01")
    p.add_argument("--pyes", type=Path, help="saved P(yes) CSV, scored instead of BATCH_OUT (see above)")
    p.add_argument("--questions", type=Path, help="with --pyes: the questions JSON with the thresholds")
    p.add_argument("--sets", type=Path, help="with --pyes: CSV with video, label and set, instead of LABELS_CSV")
    p.add_argument("--set", help="with --pyes: the set to score, e.g. train, test or other")
    args = p.parse_args()
    if args.pyes:
        if args.batch_out or not (args.questions and args.sets and args.set):
            p.error("--pyes needs --questions, --sets, --set and OUT_DIR only (no BATCH_OUT, no LABELS_CSV)")
        labels, questions, pyes = load_saved(args.pyes, args.questions, args.sets, args.set)
    elif args.labels:
        labels, questions, pyes = load(args.batch_out, args.labels, args.video_suffix)
    else:
        p.error("give BATCH_OUT LABELS_CSV OUT_DIR, or --pyes")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        as_run, _ = score_with(questions, pyes, labels)
        json.dump(as_run, open(args.out_dir / "score_as_run.json", "w"), indent=1)
        questions = calibrate(questions, pyes, labels, max_right=args.calibrate == "max_right")
        json.dump(questions, open(args.out_dir / "questions_calibrated.json", "w"), indent=1)
    s, rows = score_with(questions, pyes, labels)
    json.dump(s, open(args.out_dir / "score.json", "w"), indent=1)
    with open(args.out_dir / "pyes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{s['correct']}/{s['videos']} right, bad caught {s['bad_caught']}/{s['bad_total']}, "
          f"good passed {s['good_passed']}/{s['good_total']}, margin {s['margin']}"
          + (" (thresholds set from this batch" + (", max_right" if args.calibrate == "max_right" else "") + ")"
             if args.calibrate else ""))


if __name__ == "__main__":
    main()
