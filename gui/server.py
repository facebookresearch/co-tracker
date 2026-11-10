"""Web GUI backend: pick/upload videos, run scripts/track_video.py and scripts/compare_tracks.py
as background jobs, and serve the resulting videos to the browser.

Run inside the CoTracker container (see docker/gui.sh):
    python -m gui.server --host 0.0.0.0 --port 7860
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
GUI_OUT = REPO / "outputs" / "gui"
PREVIEWS = GUI_OUT / ".previews"
# Videos uploaded from another computer are cached here; videos on the server are read in place.
UPLOADS = GUI_OUT / ".uploads"

# Host folders mounted read-only at their own paths (MOUNTS in docker/gui.sh).
HOST_MOUNTS = [Path(m).resolve() for m in os.environ.get("HOST_MOUNTS", "").split() if Path(m).is_dir()]
# Directories the GUI may list, read and serve. /data is the optional DATA mount of docker/run.sh.
ROOTS = [REPO, Path("/data")] + HOST_MOUNTS
BROWSE_DIRS = [REPO / "source_video", REPO / "assets", Path("/data")]
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
EXTS = {"video": VIDEO_EXT, "image": IMAGE_EXT, "npy": {".npy"}, "dir": set()}

app = FastAPI()
jobs = {}
jobs_lock = threading.Lock()

# ---------------------------------------------------------------- shared grid state
# What each of the 6 blocks shows (source, result, settings, job, status), kept on the server so
# every browser and every reload sees the same thing. Each change bumps a version number;
# pages poll /api/state?since=<version> for blocks that changed.
STATE_FILE = GUI_OUT / ".gui_state.json"
SLOTS = {f"r{r}{c}" for r in (1, 2) for c in "abc"}
BLOCK_KEYS = ("src", "result", "view", "params", "showCurve", "curveYMax", "job", "status")
state_lock = threading.Lock()


def _load_state():
    try:
        st = json.load(open(STATE_FILE))
        st["blocks"] = {k: v for k, v in st.get("blocks", {}).items() if k in SLOTS}
    except (OSError, ValueError):
        return {"version": 0, "blocks": {}}
    # Jobs die with the server: anything still "running" in the file was interrupted.
    for b in st["blocks"].values():
        if (b.get("job") or {}).get("status") == "running":
            b["job"]["status"] = "lost"
            b["status"] = {"text": "The server restarted while this job was running, so it was "
                                   "stopped. Run it again.", "failed": True}
    return st


state = _load_state()


def update_block(slot: str, fn) -> int:
    """Apply fn(block_dict) to one block under the lock, bump its version, persist."""
    with state_lock:
        b = state["blocks"].setdefault(slot, {})
        fn(b)
        state["version"] = state.get("version", 0) + 1
        b["version"] = state["version"]
        GUI_OUT.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        tmp.replace(STATE_FILE)
        return b["version"]


def read_tail(log_path, lines=40) -> str:
    try:
        with open(log_path, errors="replace") as f:
            tail = f.read().replace("\r", "\n").splitlines()[-lines:]
    except OSError:
        return ""
    return "\n".join(l for l in tail if "FutureWarning" not in l and "torch.load" not in l)


def resolve(path: str) -> Path:
    """Repo-relative or absolute path, restricted to ROOTS."""
    p = Path(path)
    p = (p if p.is_absolute() else REPO / p).resolve()
    if not any(p == r or r in p.parents for r in ROOTS):
        raise HTTPException(403, f"path outside allowed folders: {path}")
    return p


def rel(p: Path) -> str:
    """Path as shown to the user / passed to the scripts (repo-relative when possible)."""
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def ffprobe(p: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_read_packets",
         "-of", "json", str(p)],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(out.stdout)["streams"][0]

    def rate(r):
        n, d = (r or "0/1").split("/")
        return float(n) / float(d) if float(d) else 0.0

    return dict(
        codec=s.get("codec_name"), pix_fmt=s.get("pix_fmt"),
        width=int(s["width"]), height=int(s["height"]),
        fps=rate(s.get("avg_frame_rate")) or rate(s.get("r_frame_rate")) or 30.0,
        frames=int(s.get("nb_read_packets") or 0),
    )


def browser_playable(p: Path, info: dict) -> bool:
    ext = p.suffix.lower()
    if ext in {".mp4", ".mov", ".m4v"}:
        return info["codec"] == "h264" and info["pix_fmt"] in {"yuv420p", "yuvj420p"}
    return ext == ".webm" and info["codec"] in {"vp8", "vp9", "av1"}


def preview_for(p: Path, info: dict) -> Path:
    """H.264 copy (1 output frame per input frame) for codecs browsers cannot play."""
    if browser_playable(p, info):
        return p
    st = p.stat()
    key = hashlib.sha1(f"{p}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()[:16]
    out = PREVIEWS / f"{key}.mp4"
    if not out.exists():
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.mp4")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(p), "-map", "0:v:0", "-vsync", "passthrough",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-an", str(tmp)],
            check=True,
        )
        tmp.rename(out)
    return out


# ---------------------------------------------------------------- files / media

@app.get("/api/files")
def list_files(kind: str = "video"):
    exts = VIDEO_EXT if kind == "video" else IMAGE_EXT
    files = []
    for d in BROWSE_DIRS:
        if not d.is_dir():
            continue
        for root, dirs, names in os.walk(d):
            dirs[:] = sorted(x for x in dirs if not x.startswith("."))
            for n in sorted(names):
                if Path(n).suffix.lower() in exts:
                    files.append(rel(Path(root) / n))
            if len(files) > 2000:
                break
    return {"files": files}


@app.get("/api/browse")
def browse(dir: str = "", kind: str = "video"):
    """One folder's subfolders and matching files. Empty dir = the start locations."""
    exts = EXTS.get(kind, VIDEO_EXT)
    if not dir:
        starts = [d for d in BROWSE_DIRS + [REPO / "outputs", REPO] + HOST_MOUNTS if d.is_dir()]
        return {"dir": "", "parent": None, "dirs": [rel(d) if d != REPO else "." for d in starts],
                "files": [], "results": []}
    d = resolve(dir)
    if not d.is_dir():
        raise HTTPException(404, f"folder not found: {dir}")
    dirs, files, results = [], [], []  # results: subfolders holding a tracking result
    for e in sorted(d.iterdir(), key=lambda e: e.name.lower()):
        if e.name.startswith("."):
            continue
        if e.is_dir():
            dirs.append(rel(e))
            if (e / "tracks.npy").is_file():
                results.append(rel(e))
        elif e.suffix.lower() in exts:
            files.append(rel(e))
    at_root = d in ROOTS
    parent = "" if at_root else (rel(d.parent) if d.parent != REPO else ".")
    return {"dir": rel(d) if d != REPO else ".", "parent": parent, "dirs": dirs, "files": files,
            "results": results}


# Extra renders that may sit next to a source video (e.g. demo_008/demo_008_depth.mp4).
MODALITIES = [("depth", "Depth"), ("geoedge", "Canny edge"), ("instance", "Segmentation"),
              ("normals", "Normals")]


@app.get("/api/siblings")
def siblings(path: str):
    """Other views of a source video found in its folder: <stem>_<kind>.<ext> first (specific to
    this video), then <demo folder>_<kind> / <stem minus its last _part>_<kind> (per demo)."""
    p = resolve(path)
    folder = p.parent
    prefixes = [p.stem]
    if folder.name.startswith("demo_"):
        prefixes.append(folder.name)
    if "_" in p.stem:
        prefixes.append(p.stem.rsplit("_", 1)[0])
    options = []
    for key, label in MODALITIES:
        found = next((f for pre in dict.fromkeys(prefixes) for ext in (".mp4", ".mov", ".webm", ".mkv", ".avi")
                      if (f := folder / f"{pre}_{key}{ext}").is_file() and f != p), None)
        if found:
            options.append({"key": key, "label": label, "path": rel(found)})
    return {"options": options}


@app.get("/api/compare_stats")
def compare_stats(path: str):
    """Per-frame difference curve of a compare run: per_frame.csv next to its compare.mp4."""
    p = resolve(path)
    csv_path = (p if p.is_dir() else p.parent) / "per_frame.csv"
    if not csv_path.is_file():
        raise HTTPException(404, "no per_frame.csv for this video")
    cols = {k: [] for k in ("frame", "mean", "median", "p95", "max", "n_valid")}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k in cols:
                v = float(row[k])
                cols[k].append(None if v != v else v)  # NaN (no valid points) -> null
    summary_path = csv_path.parent / "summary.json"
    if summary_path.is_file():
        cols["summary"] = json.load(open(summary_path))
    else:  # compare runs made before summary.json existed: everything but the overall median
        m, n = cols["mean"], cols["n_valid"]
        trap = lambda y: sum((a + b) / 2 for a, b in zip(y, y[1:]) if a is not None and b is not None)
        tot = sum(c for v, c in zip(m, n) if v is not None)
        cols["summary"] = dict(
            overall_mean=sum(v * c for v, c in zip(m, n) if v is not None) / tot if tot else None,
            overall_median=None, auc_mean=trap(m), auc_median=trap(cols["median"]),
        )
    return cols


@app.get("/api/result")
def result(dir: str):
    """An existing tracking result folder: its visualization and (if known) its source video."""
    d = resolve(dir)
    if not (d / "tracks.npy").is_file():
        raise HTTPException(400, f"not a tracking result (no tracks.npy): {dir}")
    source = None
    try:
        v = json.load(open(d / "meta.json"))["video"]
        sp = resolve(v)
        source = rel(sp) if sp.is_file() else None
    except (OSError, KeyError, ValueError, HTTPException):
        pass
    video = d / "tracks.mp4"
    return {"dir": rel(d), "video": rel(video) if video.is_file() else None, "source": source,
            "has_meta": (d / "meta.json").is_file()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "upload.mp4").name
    dest = UPLOADS / name
    stem, ext, k = dest.stem, dest.suffix, 1
    while dest.exists():
        dest = UPLOADS / f"{stem}_{k}{ext}"
        k += 1
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": rel(dest)}


@app.get("/api/meta")
def meta(path: str):
    p = resolve(path)
    if not p.is_file():
        raise HTTPException(404, f"not found: {path}")
    try:
        info = ffprobe(p)
        shown = preview_for(p, info)
    except subprocess.CalledProcessError as e:
        raise HTTPException(400, f"cannot read video {path}: {e.stderr or e}")
    return dict(info, path=rel(p), url=f"/media?path={rel(shown)}&v={int(shown.stat().st_mtime)}")


@app.get("/media")
def media(path: str):
    p = resolve(path)
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)  # supports HTTP Range, needed for seeking


@app.get("/api/gpus")
def gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"gpus": []}
    rows = [r.split(", ") for r in out.strip().splitlines()]
    return {"gpus": [dict(index=int(i), name=n, used_mb=int(u), total_mb=int(t)) for i, n, u, t in rows]}


# ---------------------------------------------------------------- state API

@app.get("/api/state")
def get_state(since: int = 0):
    with state_lock:
        return {"version": state.get("version", 0),
                "blocks": {k: v for k, v in state["blocks"].items() if v.get("version", 0) > since}}


@app.put("/api/state/{slot}")
def put_block(slot: str, body: dict = Body(...)):
    if slot not in SLOTS:
        raise HTTPException(404, f"unknown block {slot}")

    def fn(b):
        new = {k: body.get(k) for k in BLOCK_KEYS}
        # Don't let a page that hasn't seen a job finish yet overwrite the server's outcome.
        old_job, new_job = b.get("job") or {}, new.get("job") or {}
        if (old_job.get("id") and old_job.get("id") == new_job.get("id")
                and old_job.get("status") != "running" and new_job.get("status") == "running"):
            for k in ("job", "result", "view", "status"):
                new[k] = b.get(k)
        b.clear()
        b.update(new)

    return {"version": update_block(slot, fn)}


# ---------------------------------------------------------------- jobs

class TrackReq(BaseModel):
    slot: str
    video: str
    gpus: list[int] = [0]
    mode: str = "offline"
    grid_size: str = "200x100"
    radius: int = 2
    chunk_size: int = 0
    frame_stride: int = 1
    max_frames: int = 0
    grid_query_frame: int = 0
    segment_len: int = 10
    backward_tracking: bool = False
    mask: str = ""


class CompareReq(BaseModel):
    slot: str
    a: str
    b: str
    view: str = "pairs"
    metric: str = "position"
    time: str = "truncate"
    radius: int = 2
    vmax: float | None = None
    video_a: str = ""  # source video of the first result; needed when it has no meta.json
    names: list[str] = []  # display names of the two videos (first, second)
    mask: str = ""  # optional .npy mask: compare only points inside its True area
    mask_step: int = 5
    mask_rule: str = "both"


def out_dir(name: str) -> Path:
    """outputs/gui/<name>, or <name>_2, <name>_3, ... if taken (never overwrite a shown result)."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip(".") or "video"
    GUI_OUT.mkdir(parents=True, exist_ok=True)
    k = 1
    while True:
        d = GUI_OUT / (safe if k == 1 else f"{safe}_{k}")
        try:
            d.mkdir()  # atomic: two jobs can't claim the same folder
            return d
        except FileExistsError:
            k += 1


def video_stem(track_dir: Path) -> str:
    """Source video name of a tracking result (from its meta.json)."""
    try:
        return Path(json.load(open(track_dir / "meta.json"))["video"]).stem
    except (OSError, KeyError, ValueError):
        return track_dir.name


def start_job(kind: str, cmd: list[str], out: Path, result_dir: Path, result_video: Path,
              slot: str = "", result_extra: dict | None = None):
    jid = uuid.uuid4().hex[:12]
    log_path = out / "job.log"
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
    log = open(log_path, "w")
    log.write("$ " + " ".join(cmd) + "\n\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    job = dict(id=jid, kind=kind, status="running", log=str(log_path), proc=proc,
               result_dir=rel(result_dir), result_video=rel(result_video), started=time.time())
    with jobs_lock:
        jobs[jid] = job
    if slot in SLOTS:
        def started(b):
            b["job"] = {"id": jid, "status": "running"}
            b["status"] = {"text": "Starting…", "failed": False}
        update_block(slot, started)

    def wait():
        rc = proc.wait()
        log.close()
        with jobs_lock:
            status = job["status"]
        if status == "running":
            status = "done" if rc == 0 and result_video.exists() else "failed"
        # Record the outcome in the block state first, so a page that sees the job finish
        # finds the result already there.
        if slot in SLOTS:
            def finished(b):
                if (b.get("job") or {}).get("id") != jid:
                    return  # block moved on (cleared / new video) while the job ran
                b["job"] = {"id": jid, "status": status}
                if status == "done":
                    b["result"] = dict(dir=rel(result_dir), video=rel(result_video), loaded=False,
                                       **(result_extra or {}))
                    b["view"] = "result"
                    b["status"] = None
                elif status == "cancelled":
                    b["status"] = {"text": "Cancelled.", "failed": True}
                else:
                    b["status"] = {"text": f"{status.upper()} (exit {rc})\n{read_tail(log_path)}",
                                   "failed": True}
            update_block(slot, finished)
        with jobs_lock:
            job["status"] = status
            job["returncode"] = rc

    threading.Thread(target=wait, daemon=True).start()
    return {"job": jid}


@app.post("/api/track")
def track(r: TrackReq):
    video = resolve(r.video)
    if not video.is_file():
        raise HTTPException(404, f"video not found: {r.video}")
    if r.mode not in {"offline", "online"}:
        raise HTTPException(400, "mode must be offline or online")
    out = out_dir(video.stem)
    cmd = [sys.executable, "scripts/track_video.py", "--video", rel(video), "--save_dir", rel(out),
           "--mode", r.mode, "--grid_size", r.grid_size, "--radius", str(r.radius),
           "--chunk_size", str(r.chunk_size), "--frame_stride", str(r.frame_stride),
           "--max_frames", str(r.max_frames), "--grid_query_frame", str(r.grid_query_frame),
           "--segment_len", str(r.segment_len)]
    if r.gpus:
        cmd += ["--gpus", ",".join(str(g) for g in r.gpus)]
    if r.backward_tracking:
        cmd.append("--backward_tracking")
    if r.mask:
        cmd += ["--mask", rel(resolve(r.mask))]
    return start_job("track", cmd, out, out, out / "tracks.mp4", slot=r.slot)


@app.post("/api/compare")
def compare(r: CompareReq):
    a, b = resolve(r.a), resolve(r.b)
    for d in (a, b):
        if not (d / "tracks.npy").is_file():
            raise HTTPException(400, f"no tracking result in {rel(d)}")
    out = out_dir(f"{video_stem(a)}_vs_{video_stem(b)}")
    cmd = [sys.executable, "scripts/compare_tracks.py", "--a", rel(a), "--b", rel(b),
           "--out", rel(out), "--view", r.view, "--metric", r.metric, "--time", r.time,
           "--radius", str(r.radius)]
    if r.vmax:
        cmd += ["--vmax", str(r.vmax)]
    if r.video_a:
        cmd += ["--video", rel(resolve(r.video_a))]
    if r.mask:
        if r.mask_rule not in {"both", "a", "either"}:
            raise HTTPException(400, "mask_rule must be both, a or either")
        cmd += ["--mask", rel(resolve(r.mask)), "--mask_step", str(max(1, r.mask_step)),
                "--mask_rule", r.mask_rule]
    return start_job("compare", cmd, out, out, out / "compare.mp4", slot=r.slot,
                     result_extra={"names": r.names})


@app.get("/api/jobs/{jid}")
def job_status(jid: str, lines: int = 40):
    with jobs_lock:
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404)
        info = {k: v for k, v in job.items() if k != "proc"}
    info["tail"] = read_tail(info["log"], lines)
    info["elapsed"] = round(time.time() - info["started"], 1)
    return info


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    with jobs_lock:
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404)
        if job["status"] == "running":
            job["status"] = "cancelled"
            try:
                os.killpg(job["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    return {"ok": True}


# ---------------------------------------------------------------- batch pipeline
# One batch at a time: scripts/batch_pipeline.py runs as a job; its progress file
# (<out>/batch_status.json) and a pointer in outputs/gui/.batch.json let any page show it.
BATCH_FILE = GUI_OUT / ".batch.json"


class BatchReq(BaseModel):
    root: str
    a_suffix: str = "_source"
    b_suffix: str = "_01"
    instance_suffix: str = "_instance"
    steps: dict[str, str] = {"track_a": "run", "track_b": "run", "mask": "run", "compare": "run", "vlm": "run"}
    track: dict = {}
    mask: dict = {}
    compare: dict = {}
    vlm: dict = {}


def batch_out(root: Path) -> Path:
    return GUI_OUT / root.name  # outputs/gui/<root folder name>/demo_*/...


@app.get("/api/batch/scan")
def batch_scan(root: str, a_suffix: str = "_source", b_suffix: str = "_01",
               instance_suffix: str = "_instance"):
    r = resolve(root)
    if not r.is_dir():
        raise HTTPException(404, f"folder not found: {root}")
    demos = []
    for d in sorted(x for x in r.iterdir() if x.is_dir() and x.name.startswith("demo_")):
        has = lambda suf: (d / f"{d.name}{suf}.mp4").is_file()
        demos.append(dict(name=d.name, a=has(a_suffix), b=has(b_suffix), instance=has(instance_suffix)))
    return {"root": rel(r), "out": rel(batch_out(r)), "demos": demos}


def current_batch():
    try:
        info = json.load(open(BATCH_FILE))
    except (OSError, ValueError):
        return None
    with jobs_lock:
        job = jobs.get(info.get("job"))
        info["job_status"] = job["status"] if job else "lost"  # unknown job: server restarted
    try:
        info["progress"] = json.load(open(REPO / info["out"] / "batch_status.json"))
    except (OSError, ValueError):
        info["progress"] = None
    # Difference statistics of each finished comparison (from its summary.json), for ranking.
    for steps in ((info["progress"] or {}).get("demos") or {}).values():
        c = steps.get("compare") or {}
        if c.get("dir") and c.get("status") in ("done", "skipped"):
            try:
                sm = json.load(open(REPO / c["dir"] / "summary.json"))
                for k in ("overall_mean", "overall_median", "auc_mean", "auc_median"):
                    c[k] = sm.get(k)
            except (OSError, ValueError):
                pass
        # VLM check of video B (<demo>_vlm.json): anomaly score and each question's P(yes) / Y/N answer.
        v = steps.get("vlm") or {}
        if v.get("path") and v.get("status") in ("done", "skipped"):
            try:
                res = json.load(open(REPO / v["path"]))
                v["vlm_score"] = res.get("score")
                v["answers"] = res.get("answers")
                # one column per question id, over all results (older ones may have other questions)
                known = {q["id"] for q in info.setdefault("vlm_questions", [])}
                info["vlm_questions"] += [q for q in res.get("questions") or [] if q["id"] not in known]
            except (OSError, ValueError):
                pass
    if info["job_status"] == "lost" and info["progress"] and info["progress"].get("finished"):
        info["job_status"] = "done"  # finished before the restart
    info["tail"] = read_tail(REPO / info["out"] / "job.log", 15)
    return info


@app.get("/api/batch")
def batch_get():
    return {"batch": current_batch()}


@app.post("/api/batch/start")
def batch_start(r: BatchReq):
    cur = current_batch()
    if cur and cur["job_status"] == "running":
        raise HTTPException(409, "a batch is already running; cancel it first")
    root = resolve(r.root)
    if not root.is_dir():
        raise HTTPException(404, f"folder not found: {r.root}")
    steps = ("track_a", "track_b", "mask", "compare", "vlm")
    if set(r.steps) - set(steps) or set(r.steps.values()) - {"run", "missing", "off"}:
        raise HTTPException(400, "steps: {track_a|track_b|mask|compare|vlm: run|missing|off}")
    if all(r.steps.get(k, "run") == "off" for k in steps):
        raise HTTPException(400, "all steps are off")
    out = batch_out(root)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(r.model_dump(), root=rel(root), out=rel(out))
    sys.path.insert(0, str(REPO / "scripts"))
    from extract_mask import parse_colors
    for key, spec in (("mask colors", cfg["mask"].get("colors")), ("towel color", cfg["vlm"].get("towel_color"))):
        if spec:
            try:
                parse_colors(spec)
            except ValueError as e:
                raise HTTPException(400, f"{key}: {e}")
    with open(out / "batch_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    (out / "batch_status.json").unlink(missing_ok=True)  # don't show the previous run's progress
    res = start_job("batch", [sys.executable, "scripts/batch_pipeline.py", "--config",
                              rel(out / "batch_config.json")], out, out, out / "batch_status.json")
    with open(BATCH_FILE, "w") as f:
        json.dump({"job": res["job"], "root": rel(root), "out": rel(out), "started": time.time(),
                   "config": cfg}, f, indent=1)
    return res


@app.post("/api/batch/cancel")
def batch_cancel():
    cur = current_batch()
    if not cur:
        raise HTTPException(404, "no batch")
    return cancel(cur["job"])


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    print(f"CoTracker GUI: open http://localhost:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
