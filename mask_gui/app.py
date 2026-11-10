"""
Instance Mask Picker
=====================
A small Flask GUI for loading a color-coded instance-segmentation video
(e.g. demo_008_instance.mp4), clicking on one or more colored blocks to
select the instances you want, combining them into a single mask, saving
the result to disk, and letting you scrub through the produced mask to
check it.

The input videos are flat, solid-color instance maps: each instance keeps
the same color in every frame. So no frame-to-frame tracking is needed --
a mask is just "which pixels are close enough to one of the selected
colors", computed independently per frame.

Everything lives under DATA_DIR (default /data, mounted from the host via
docker-compose). Videos are read from any *.mp4 found anywhere under
DATA_DIR (so you can pick any video in the mounted project, not just ones
at the top level), outputs are written to
DATA_DIR/output/<video_path_flattened>/<mask_id>/.
"""
import json
import io
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-memory caches (kept simple: single-process, single-user tool)
# ---------------------------------------------------------------------------
_video_cache = {}  # video_name -> {"frames": [np.ndarray BGR, ...], "fps": float}
_mask_cache = {}    # (video_name, mask_id) -> {"masks": [np.ndarray bool, ...], ...}


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def list_videos():
    """Every *.mp4 found anywhere under DATA_DIR (excluding our own output
    folder), as paths relative to DATA_DIR -- so the user can pick any video
    in the mounted project, not just ones dropped at the top level."""
    videos = []
    for p in DATA_DIR.rglob("*.mp4"):
        if OUTPUT_DIR == p or OUTPUT_DIR in p.parents:
            continue
        videos.append(p.relative_to(DATA_DIR).as_posix())
    return sorted(videos)


def list_npy_files():
    """Every *.npy found anywhere under DATA_DIR -- includes masks this tool
    already saved under output/, as well as anything written by outside
    tools (e.g. the batch_generate_masks.py script), so either can be
    loaded back in for a visual check."""
    return sorted(p.relative_to(DATA_DIR).as_posix() for p in DATA_DIR.rglob("*.npy"))


def resolve_under_data_dir(rel_path):
    """Resolve a user-supplied path relative to DATA_DIR, refusing anything
    that escapes it (e.g. via '..')."""
    data_dir = DATA_DIR.resolve()
    candidate = (DATA_DIR / rel_path).resolve()
    if candidate != data_dir and data_dir not in candidate.parents:
        abort(400, "invalid path")
    return candidate


def load_video(name):
    if name in _video_cache:
        return _video_cache[name]
    path = DATA_DIR / name
    if not path.exists():
        abort(404, f"video not found: {name}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        abort(400, f"could not open video: {name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        abort(400, f"could not read any frames from {name}")
    entry = {"frames": frames, "fps": fps}
    _video_cache[name] = entry
    return entry


def encode_jpeg(img_bgr, quality=92):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        abort(500, "jpeg encode failed")
    return buf.tobytes()


def send_bgr_as_jpeg(img):
    return send_file(io.BytesIO(encode_jpeg(img)), mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# Mask building: each instance is one flat color held for the whole video,
# so the mask for a set of selected colors is a simple per-frame threshold.
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


def build_masks(frames, colors_bgr, tolerance, use_cleanup=True, invert=False):
    """For every frame, union together the pixels matching any of the
    selected colors (within `tolerance`) into a single boolean mask. If
    `invert` is set, the mask is flipped to everything *except* those
    colors (e.g. to grab the background, or "all the other instances")."""
    masks = []
    for frame in frames:
        m = np.zeros(frame.shape[:2], dtype=bool)
        for color_bgr in colors_bgr:
            m |= color_distance(frame, color_bgr) < tolerance
        if invert:
            m = ~m
        if use_cleanup:
            m = clean_mask(m)
        masks.append(m)
    return masks


# ---------------------------------------------------------------------------
# Persistence -- just the mask stack (.npy) plus its metadata. The frame
# previews shown in the GUI are rendered on the fly from this, so there's no
# need to also keep a PNG-per-frame copy or an overlay video on disk.
# ---------------------------------------------------------------------------
def video_key(video):
    """Collision-safe directory name for a (possibly nested) video path."""
    return video.replace("/", "__")


def masks_root_for(video):
    return OUTPUT_DIR / video_key(video)


def mask_dir_for(video, mask_id):
    return masks_root_for(video) / mask_id


def save_mask_to_disk(video, mask_id, frames, masks, colors_bgr, tolerance, cleanup, invert):
    out_dir = mask_dir_for(video, mask_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    stack = np.zeros((len(masks), h, w), dtype=bool)
    for i, m in enumerate(masks):
        stack[i] = m
    np.save(out_dir / "mask_stack.npy", stack)

    meta = {
        "video": video,
        "mask_id": mask_id,
        "colors_bgr": [list(c) for c in colors_bgr],
        "colors_rgb": [[int(c[2]), int(c[1]), int(c[0])] for c in colors_bgr],
        "tolerance": tolerance,
        "cleanup": cleanup,
        "invert": invert,
        "num_frames": len(masks),
        "found_frames": int(sum(1 for m in masks if m.sum() > 0)),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def save_loaded_mask_to_disk(video, mask_id, masks, source_npy):
    """Persist a mask that came from an externally-supplied .npy file (not
    generated by a color pick), so it shows up in the saved-masks list and
    can be previewed/downloaded the same way as any other mask."""
    out_dir = mask_dir_for(video, mask_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    stack = np.stack(masks, axis=0)
    np.save(out_dir / "mask_stack.npy", stack)

    meta = {
        "video": video,
        "mask_id": mask_id,
        "colors_bgr": [],
        "colors_rgb": [],
        "source_npy": source_npy,
        "num_frames": len(masks),
        "found_frames": int(sum(1 for m in masks if m.sum() > 0)),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_mask_from_disk(video, mask_id):
    out_dir = mask_dir_for(video, mask_id)
    meta_path = out_dir / "meta.json"
    npy_path = out_dir / "mask_stack.npy"
    if not meta_path.exists() or not npy_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    stack = np.load(npy_path)
    masks = [stack[i] for i in range(stack.shape[0])]
    entry = {"masks": masks, "colors_bgr": [tuple(c) for c in meta["colors_bgr"]], "tolerance": meta["tolerance"]}
    _mask_cache[(video, mask_id)] = entry
    return entry


def get_mask(video, mask_id):
    key = (video, mask_id)
    if key in _mask_cache:
        return _mask_cache[key]
    return load_mask_from_disk(video, mask_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/videos")
def api_videos():
    return jsonify({"videos": list_videos()})


@app.route("/api/load", methods=["POST"])
def api_load():
    name = (request.get_json() or {}).get("video")
    if not name:
        abort(400, "missing 'video'")
    entry = load_video(name)
    h, w = entry["frames"][0].shape[:2]
    return jsonify({"num_frames": len(entry["frames"]), "width": w, "height": h, "fps": entry["fps"]})


@app.route("/api/frame/<path:video>/<int:idx>")
def api_frame(video, idx):
    entry = load_video(video)
    if not (0 <= idx < len(entry["frames"])):
        abort(404)
    return send_bgr_as_jpeg(entry["frames"][idx])


@app.route("/api/sample_color", methods=["POST"])
def api_sample_color():
    """Return the exact pixel color at (frame_idx, x, y) from the decoded
    video frame (not the display JPEG), so clicking always samples the
    real instance color."""
    data = request.get_json() or {}
    video = data.get("video")
    if not video:
        abort(400, "missing 'video'")
    frame_idx = int(data.get("frame_idx", 0))
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))

    entry = load_video(video)
    frames = entry["frames"]
    h, w = frames[0].shape[:2]
    if not (0 <= frame_idx < len(frames)) or not (0 <= x < w and 0 <= y < h):
        abort(400, "point/frame out of bounds")

    b, g, r = frames[frame_idx][y, x].tolist()
    return jsonify({"rgb": [int(r), int(g), int(b)]})


@app.route("/api/make_mask", methods=["POST"])
def api_make_mask():
    """Combine every selected color into one mask, computed per frame, and
    save it to disk."""
    data = request.get_json() or {}
    video = data.get("video")
    colors = data.get("colors")  # list of [r, g, b]
    tolerance = float(data.get("tolerance", 30))
    cleanup = bool(data.get("cleanup", True))
    invert = bool(data.get("invert", False))
    if not video:
        abort(400, "missing 'video'")
    if not colors:
        abort(400, "need at least one selected color")

    entry = load_video(video)
    frames = entry["frames"]
    colors_bgr = [(int(c[2]), int(c[1]), int(c[0])) for c in colors]

    masks = build_masks(frames, colors_bgr, tolerance, use_cleanup=cleanup, invert=invert)

    mask_id = uuid.uuid4().hex[:8]
    _mask_cache[(video, mask_id)] = {"masks": masks, "colors_bgr": colors_bgr, "tolerance": tolerance}
    meta = save_mask_to_disk(video, mask_id, frames, masks, colors_bgr, tolerance, cleanup, invert)

    return jsonify({
        "mask_id": mask_id,
        "colors_rgb": meta["colors_rgb"],
        "num_frames": meta["num_frames"],
        "found_frames": meta["found_frames"],
    })


@app.route("/api/npy_files")
def api_npy_files():
    return jsonify({"files": list_npy_files()})


@app.route("/api/load_mask_npy", methods=["POST"])
def api_load_mask_npy():
    """Load an existing .npy mask file (e.g. written by the batch script,
    or downloaded earlier) and register it as a mask for `video` so it can
    be previewed/downloaded through the normal mask UI."""
    data = request.get_json() or {}
    video = data.get("video")
    npy_rel = data.get("npy_path")
    if not video:
        abort(400, "missing 'video'")
    if not npy_rel:
        abort(400, "missing 'npy_path'")

    npy_path = resolve_under_data_dir(npy_rel)
    if not npy_path.exists() or npy_path.suffix != ".npy":
        abort(404, "npy file not found")

    entry = load_video(video)
    frames = entry["frames"]
    n = len(frames)
    h, w = frames[0].shape[:2]

    try:
        arr = np.load(npy_path)
    except Exception as e:
        abort(400, f"could not load npy: {e}")

    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        abort(400, f"expected a 3D (frames, H, W) array, got shape {list(arr.shape)}")
    if arr.shape[0] == 1 and n > 1:
        arr = np.repeat(arr, n, axis=0)  # a single-frame mask -> hold it for every frame
    if tuple(arr.shape) != (n, h, w):
        abort(400, f"shape mismatch: npy is {list(arr.shape)}, loaded video is [{n}, {h}, {w}]")

    masks = [arr[i].astype(bool) for i in range(arr.shape[0])]

    mask_id = uuid.uuid4().hex[:8]
    _mask_cache[(video, mask_id)] = {"masks": masks, "colors_bgr": [], "tolerance": None}
    meta = save_loaded_mask_to_disk(video, mask_id, masks, npy_rel)

    return jsonify({
        "mask_id": mask_id,
        "num_frames": meta["num_frames"],
        "found_frames": meta["found_frames"],
        "source_npy": npy_rel,
    })


@app.route("/api/masks/<path:video>")
def api_masks(video):
    out_dir = masks_root_for(video)
    result = []
    if out_dir.exists():
        for d in sorted(out_dir.iterdir()):
            meta_path = d / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    result.append(json.load(f))
    return jsonify({"masks": result})


@app.route("/api/mask_frame/<path:video>/<mask_id>/<int:idx>")
def api_mask_frame(video, mask_id, idx):
    mode = request.args.get("mode", "overlay")
    entry = get_mask(video, mask_id)
    if entry is None:
        abort(404, "mask not found")
    frames = load_video(video)["frames"]
    if not (0 <= idx < len(frames)):
        abort(404)
    frame = frames[idx]
    mask = entry["masks"][idx] if idx < len(entry["masks"]) else None
    mask = mask if mask is not None else np.zeros(frame.shape[:2], dtype=bool)

    if mode == "raw":
        img = frame
    elif mode == "mask":
        img = np.zeros_like(frame)
        img[mask] = (255, 255, 255)
    else:  # overlay
        img = frame.copy()
        if mask.any():
            img[mask] = (0.4 * img[mask].astype(np.float32) + 0.6 * np.array([0, 0, 255], dtype=np.float32)).astype(np.uint8)
    return send_bgr_as_jpeg(img)


@app.route("/api/download/<path:video>/<mask_id>")
def api_download(video, mask_id):
    npy_path = mask_dir_for(video, mask_id) / "mask_stack.npy"
    if not npy_path.exists():
        abort(404)
    stem = Path(video).stem
    return send_file(npy_path, mimetype="application/octet-stream", as_attachment=True,
                      download_name=f"{stem}_{mask_id}_mask.npy")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
