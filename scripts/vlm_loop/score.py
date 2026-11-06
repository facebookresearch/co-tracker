"""Score one VLM batch against the labels of the set.

    python scripts/vlm_loop/score.py BATCH_OUT LABELS_CSV OUT_DIR [--calibrate [max_right]] [--video_suffix _01]
    python scripts/vlm_loop/score.py --pyes PYES_CSV --questions QUESTIONS_JSON --sets SETS_CSV --set SET OUT_DIR \
        [--calibrate [max_right]]
    python scripts/vlm_loop/score.py BATCH_OUT

BATCH_OUT is the GUI output folder of the batch (outputs/gui/<batch name>), with <demo>/<demo>_01_vlm.json.
A video is marked bad when any question is Yes (P(yes) > the question's threshold).

BATCH_OUT alone checks a GUI batch on a folder of a share folder (for example outputs/gui/O after a batch on
<share>/videos/O), with the videos the GUI table shows answers for: those whose VLM step is done or skipped in
this batch (batch_status.json). The label of a video is the review of the current video of its demo in that share
folder (<share>/lists/videos.csv and lists/review.csv; approved and weak = good, rejected = bad), and the
thresholds are the ones the questions were run with, as in the GUI table. Nothing is written: it prints the score
line, the flagged videos, the number of Yes of every question, and the videos left out (no VLM result in this
batch, for example failed, cancelled or an unreadable result file; no review; or a result of a video in another
folder, kept from an earlier batch on another share folder: the GUI writes every batch on a folder named O to
outputs/gui/O). It stops when the videos have results of different question sets, or results of the first VLM
check (Y/N without P(yes)), kept by the GUI step modes "If missing" or "Off": run the batch again with the VLM
step on "Run". A video made again keeps its file name, so the result file does not show which video the batch
saw: when the share folder was updated after the batch started (its README.md is newer), a warning is printed,
and the batch should run again. A relative root in batch_config.json (a share folder inside the repo) is taken
from the repo of this score.py.

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

Writes (not with BATCH_OUT alone):
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
import time
from pathlib import Path

SHARE_LABEL = {"approved": "good", "weak": "good", "rejected": "bad"}


def load(batch_out: Path, labels_csv: Path, video_suffix: str) -> tuple[dict, list[dict], dict]:
    """-> (labels {video: good|bad}, questions as run, P(yes) {video: {question id: p}})."""
    with open(labels_csv, newline="") as f:
        labels = {r["video"]: r["label"] for r in csv.DictReader(f)}
    return (labels, *load_results(batch_out, labels, video_suffix))


def load_results(batch_out: Path, videos, video_suffix: str) -> tuple[list[dict], dict]:
    """-> (questions as run, P(yes) {video: {question id: p}}) of the videos, from the GUI batch output folder."""
    pyes, questions = {}, None
    for v in sorted(videos):
        res = json.load(open(batch_out / v / f"{v}{video_suffix}_vlm.json"))
        pyes[v] = {q: a["p_yes"] for q, a in res["answers"].items()}
        questions = questions or res["questions"]
    return questions, pyes


def share_verdicts(lists: Path) -> dict:
    """{demo number: review verdict of its current video} from the list files of a share folder (the rule of the
    share folder: videos.csv gives the current video, review.csv its review)."""
    with open(lists / "review.csv", newline="") as f:
        reviews = {r["video"]: r["verdict"] for r in csv.DictReader(f)}
    with open(lists / "videos.csv", newline="") as f:
        return {int(r["demo"]): reviews.get(f"{r['cosmos_run']}/{r['name']}", r.get("verdict", ""))
                for r in csv.DictReader(f) if r["where"] == "current"}


def read_result(path: Path) -> dict | None:
    """A <demo>_01_vlm.json, or None when it cannot be read (the GUI then shows no answers for it)."""
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def gui_batch(batch_out: Path) -> tuple[dict, list[dict], dict, dict, float | None]:
    """BATCH_OUT alone: -> (labels {video: good|bad}, questions as run, P(yes) {video: {question id: p}}, the videos
    left out {why: [videos]}, the time of a share update after the batch started or None) of a GUI batch on a
    folder of a share folder (see BATCH_OUT alone)."""
    try:
        cfg = json.load(open(batch_out / "batch_config.json"))
        status = json.load(open(batch_out / "batch_status.json"))
    except OSError as e:
        raise SystemExit(f"{batch_out} is not the output folder of a GUI batch ({e})")
    root = Path(cfg["root"])
    if not root.is_absolute():  # the GUI writes repo-relative paths when it can
        root = Path(__file__).resolve().parents[2] / root
    share = root.parent.parent
    if not ((share / "lists" / "review.csv").is_file() and (share / "lists" / "videos.csv").is_file()):
        raise SystemExit(f"the batch ran on {root}, not on a folder of a share folder (no review.csv and videos.csv "
                         f"in {share / 'lists'}); give BATCH_OUT LABELS_CSV OUT_DIR")
    verdict = share_verdicts(share / "lists")
    left = {"no VLM result in this batch": [], "result of a video in another folder": [], "no review": []}
    labels, results = {}, {}
    for d in sorted(status["demos"]):
        vlm = status["demos"][d].get("vlm") or {}
        res = read_result(batch_out / d / Path(vlm["path"]).name) if vlm.get("path") else None
        label = SHARE_LABEL.get(verdict.get(int(d.split("_")[1]), ""))
        if vlm.get("status") not in ("done", "skipped") or res is None:
            left["no VLM result in this batch"].append(d)
        elif Path(res.get("video", "")).resolve().parent.parent != root.resolve():
            left["result of a video in another folder"].append(d)
        elif label is None:
            left["no review"].append(d)
        else:
            labels[d], results[d] = label, res
    readme = share / "README.md"  # share.py writes it at every update
    updated = readme.stat().st_mtime if readme.is_file() else None
    updated = updated if updated and status.get("started") and updated > status["started"] else None
    old = [d for d, res in results.items() if any("p_yes" not in a for a in res["answers"].values())]
    if old:
        raise SystemExit(f"{batch_out} has results of the first VLM check (Y/N without P(yes)) for {len(old)} of "
                         f"{len(results)} videos, e.g. {old[0]}; run the batch again with the VLM step on Run")
    sets = {}
    for d, res in results.items():
        sets.setdefault(json.dumps(res["questions"], sort_keys=True), []).append(d)
    if len(sets) > 1:
        raise SystemExit(f"the videos of {batch_out} have results of {len(sets)} different question sets ("
                         + "; ".join(f"{len(v)} video{'s' * (len(v) > 1)}, e.g. {v[0]}" for v in sets.values())
                         + "); run the batch again with the VLM step on Run")
    questions = next(iter(results.values()))["questions"] if results else []
    pyes = {d: {q: a["p_yes"] for q, a in res["answers"].items()} for d, res in results.items()}
    return labels, questions, pyes, left, updated


def print_batch(batch_out: Path) -> None:
    """BATCH_OUT alone: the score of a GUI batch on a folder of a share folder, printed (nothing is written)."""
    labels, questions, pyes, left, updated = gui_batch(batch_out)
    if not labels:
        raise SystemExit(f"no video of {batch_out} can be scored; left out: "
                         + ", ".join(f"{why} {len(v)}" for why, v in left.items() if v))
    s, _ = score_with(questions, pyes, labels)
    yes = s["yes_per_question"]
    flagged = sorted({v for q in yes.values() for v in q["good"] + q["bad"]})
    print(f"{s['correct']}/{s['videos']} (bad {s['bad_caught']}/{s['bad_total']})")
    print(f"good passed {s['good_passed']}/{s['good_total']} | flagged {len(flagged)}: {', '.join(flagged)}")
    width = max(len(q["id"]) for q in questions)
    for q in questions:
        y = yes[q["id"]]
        print(f"{q['id']:<{width}}  Yes {len(y['good']) + len(y['bad'])} (good {len(y['good'])}, bad {len(y['bad'])})")
    for why, videos in left.items():
        if videos:
            print(f"left out, {why}: {len(videos)} ({', '.join(videos)})")
    if updated:
        print(f"warning: the share folder was updated after this batch started "
              f"({time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(updated))}); a video made again since then is "
              "not found, so run the batch again to be sure")


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
    p.add_argument("paths", type=Path, nargs="+", metavar="PATH",
                   help="BATCH_OUT LABELS_CSV OUT_DIR; BATCH_OUT alone (see above); or OUT_DIR with --pyes")
    p.add_argument("--calibrate", nargs="?", const=True, choices=["max_right"],
                   help="set the thresholds from this batch's P(yes); max_right: the max_right rule (see above)")
    p.add_argument("--video_suffix", default="_01")
    p.add_argument("--pyes", type=Path, help="saved P(yes) CSV, scored instead of BATCH_OUT (see above)")
    p.add_argument("--questions", type=Path, help="with --pyes: the questions JSON with the thresholds")
    p.add_argument("--sets", type=Path, help="with --pyes: CSV with video, label and set, instead of LABELS_CSV")
    p.add_argument("--set", help="with --pyes: the set to score, e.g. train, test or other")
    args = p.parse_args()
    if args.pyes:
        if len(args.paths) != 1 or not (args.questions and args.sets and args.set):
            p.error("--pyes needs --questions, --sets, --set and OUT_DIR only (no BATCH_OUT, no LABELS_CSV)")
        out_dir = args.paths[0]
        labels, questions, pyes = load_saved(args.pyes, args.questions, args.sets, args.set)
    elif len(args.paths) == 3:
        batch_out, labels_csv, out_dir = args.paths
        labels, questions, pyes = load(batch_out, labels_csv, args.video_suffix)
    elif len(args.paths) == 1:
        if args.questions or args.sets or args.set:
            p.error("--questions, --sets and --set go with --pyes")
        if args.calibrate:
            p.error("--calibrate needs BATCH_OUT LABELS_CSV OUT_DIR")
        print_batch(args.paths[0])
        return
    else:
        p.error("give BATCH_OUT LABELS_CSV OUT_DIR, BATCH_OUT alone, or --pyes ... OUT_DIR")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        as_run, _ = score_with(questions, pyes, labels)
        json.dump(as_run, open(out_dir / "score_as_run.json", "w"), indent=1)
        questions = calibrate(questions, pyes, labels, max_right=args.calibrate == "max_right")
        json.dump(questions, open(out_dir / "questions_calibrated.json", "w"), indent=1)
    s, rows = score_with(questions, pyes, labels)
    json.dump(s, open(out_dir / "score.json", "w"), indent=1)
    with open(out_dir / "pyes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{s['correct']}/{s['videos']} right, bad caught {s['bad_caught']}/{s['bad_total']}, "
          f"good passed {s['good_passed']}/{s['good_total']}, margin {s['margin']}"
          + (" (thresholds set from this batch" + (", max_right" if args.calibrate == "max_right" else "") + ")"
             if args.calibrate else ""))


if __name__ == "__main__":
    main()
