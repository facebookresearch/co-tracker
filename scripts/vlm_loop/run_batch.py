"""Run one VLM-only batch for a VLM question loop, on the videos of one set, read straight from the share folder.

    python scripts/vlm_loop/run_batch.py --share SHARE_DIR --sets SETS_CSV --set SET --name BATCH_NAME \
        --questions QUESTIONS_JSON [--gpus 0] [--towel_color "R,G,B; R,G,B"] [--min_free_gb 18]

The videos are the demo folders of the set (the rows of --set in SETS_CSV, made by make_set.py) in the share
folder; no link or input folder is made. The script runs the VLM step of the batch pipeline
(scripts/batch_pipeline.py, used as it is: merged, Status, process and VlmCheck) on them one after the other, in
this process, as a GUI batch with all CoTracker steps off and only "vlm" on does. The results are in
outputs/gui/BATCH_NAME, in the layout of a GUI batch (batch_config.json, batch_status.json,
<demo>/<demo>_01_vlm.json, <demo>/vlm_frames/). The GUI is not used, so the batch does not show in the GUI.

Waiting for a GPU: the checker needs a GPU of --gpus with --min_free_gb free. While there is none (other jobs
use the GPUs), the script does not start and checks again every --wait_minutes. When the checker still loads on
the CPU (the space was taken between the check and the load), it waits again. After --max_wait_hours of waiting
it fails. It also fails when a video fails. A finished batch of the same name and the same questions is not run
again (the script only prints its folder).
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import batch_pipeline  # noqa: E402
import vlm_check  # noqa: E402
from make_set import set_folders  # noqa: E402


def free_gpu_gb(gpus: list[int]) -> dict[int, float]:
    """{GPU index: free GB} for the given GPUs (nvidia-smi index order, as the batch pipeline uses)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    free = {int(i): int(f) / 1024 for i, f in (line.split(",") for line in out.strip().splitlines())}
    return {g: free[g] for g in gpus if g in free}


def same_finished(out: Path, questions: list, video_suffix: str) -> bool:
    """True when out has a finished batch whose results were made with these questions (as vlm_check
    records them in every <demo><video_suffix>_vlm.json)."""
    try:
        status = json.load(open(out / "batch_status.json"))
        results = sorted(out.glob(f"demo_*/demo_*{video_suffix}_vlm.json"))
        used = [json.load(open(p)).get("questions") for p in results]
    except (OSError, ValueError):
        return False
    return bool(status.get("finished")) and bool(used) and all(u == questions for u in used)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--share", type=Path, required=True, help="the share folder with the videos (share.py)")
    p.add_argument("--sets", type=Path, required=True, help="the sets CSV (video, label, set; make_set.py)")
    p.add_argument("--set", required=True, help="the set to run, e.g. train")
    p.add_argument("--name", required=True, help="batch name: the output folder name")
    p.add_argument("--questions", required=True, help="questions JSON, relative to the repo")
    p.add_argument("--gpus", default="0", help="GPUs the checker may use, e.g. 0 or 2,3")
    p.add_argument("--towel_color", default="167,18,135")
    p.add_argument("--min_free_gb", type=float, default=18)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--video_suffix", default="_01")
    p.add_argument("--instance_suffix", default="_instance_all")
    p.add_argument("--wait_minutes", type=float, default=5, help="minutes between checks while waiting for a GPU")
    p.add_argument("--max_wait_hours", type=float, default=12, help="fail after waiting this long for a GPU")
    args = p.parse_args()

    try:  # stop here on a bad questions file, before a GPU is used
        questions = [q.to_json() for q in vlm_check.load_questions(REPO / args.questions)]
    except (RuntimeError, ValueError) as e:
        sys.exit(f"bad questions file {args.questions}: {e}")
    demos = set_folders(args.share, args.sets, args.set)
    out = REPO / "outputs" / "gui" / args.name
    if same_finished(out, questions, args.video_suffix):
        print(f"already done: {out}")
        return
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    cfg = batch_pipeline.merged({
        "root": str(args.share.resolve() / "videos"), "out": str(out.relative_to(REPO)),
        "a_suffix": "_source_all", "b_suffix": args.video_suffix, "instance_suffix": args.instance_suffix,
        "steps": {"track_a": "off", "track_b": "off", "mask": "off", "compare": "off", "vlm": "run"},
        "track": {"gpus": gpus},
        "vlm": {"questions": args.questions, "model": args.model, "towel_color": args.towel_color,
                "min_free_gb": args.min_free_gb}})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "batch_config.json", "w") as f:
        json.dump(dict(cfg, share=str(args.share), sets=str(args.sets), set=args.set), f, indent=2)
    deadline = time.time() + args.max_wait_hours * 3600
    while True:  # wait for a free GPU, load the checker there, and wait again when it lands on the CPU
        free = free_gpu_gb(gpus)
        if any(f >= args.min_free_gb for f in free.values()):
            status = batch_pipeline.Status(out / "batch_status.json", sorted(demos))
            vlm = batch_pipeline.VlmCheck(cfg, status)
            try:
                vlm.load()
            except Exception as e:  # noqa: BLE001 - the same message as the batch pipeline gives
                sys.exit(f"could not start the VLM check: {e}")
            if not vlm.where.startswith("CPU"):
                break
            vlm = None
            gc.collect()
            why = "the checker loaded on the CPU"
        else:
            why = ", ".join(f"GPU {g} has {f:.1f} GB free" for g, f in free.items()) or f"GPU {args.gpus} not found"
        if time.time() > deadline:
            sys.exit(f"no GPU of {args.gpus} with {args.min_free_gb:g} GB free after {args.max_wait_hours:g} h "
                     "of waiting")
        print(f"waiting for a GPU: {why}; need {args.min_free_gb:g} GB; next check in {args.wait_minutes:g} min",
              flush=True)
        time.sleep(args.wait_minutes * 60)
    print(f"batch {args.name}: {len(demos)} videos of set {args.set} from {args.share}", flush=True)
    t0 = time.time()
    for i, name in enumerate(sorted(demos), 1):
        print(f"[{i}/{len(demos)}] {name}", flush=True)
        batch_pipeline.process(demos[name], out / name, cfg, status)
        vlm(demos[name], out / name)
    status.data["finished"], status.data["current"] = time.time(), None
    status.save()
    failed = [d for d, st in status.data["demos"].items() if st["vlm"].get("status") not in ("done", "skipped")]
    if failed:
        sys.exit(f"batch {args.name}: the VLM check did not finish for {', '.join(failed)}")
    print(f"done in {time.time() - t0:.0f} s: {out}")


if __name__ == "__main__":
    main()
