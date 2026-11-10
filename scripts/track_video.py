"""Run CoTracker3 on a video and save tracks + a visualization.

Examples:
    # 20x20 grid on frame 0, offline model
    python scripts/track_video.py --video assets/apple.mp4 --grid_size 20

    # 100 columns x 50 rows, tiny dots
    python scripts/track_video.py --video assets/apple.mp4 --grid_size 100x50 --radius 1

    # Grid restricted to a segmentation mask, tracked both directions from frame 10
    python scripts/track_video.py --video v.mp4 --grid_size 30 --mask m.png \
        --grid_query_frame 10 --backward_tracking

    # Specific points: text file with one "t x y" per line (frame idx, pixel coords)
    python scripts/track_video.py --video v.mp4 --queries pts.txt

    # Long video: sliding-window online model (constant memory)
    python scripts/track_video.py --video long.mp4 --mode online --grid_size 20

    # OOM: split the points over 4 GPUs, at most 500 points per forward pass
    python scripts/track_video.py --video v.mp4 --grid_size 60 --gpus 0,1,2,3 --chunk_size 500

Outputs (in --out_dir/<video name>/, or exactly --save_dir if given):
    tracks.npy      float32 (T, N, 2)  pixel (x, y) per frame, original resolution. With segments
                                       (--segment_len, default 10) point i restarts at its grid
                                       position on every segment's first frame.
    visibility.npy  bool    (T, N)     whether each point is visible in each frame
    queries.npy     float32 (N, 3)     (t, x, y) query point of each track (t relative to the
                                       segment start when segments are used)
    meta.json       video path, width, height, frames, frame_stride, fps
    tracks.mp4      visualization
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer


def load_video(path, max_frames, stride):
    frames = iio.imread(path, plugin="FFMPEG")  # (T, H, W, 3) uint8
    frames = frames[::stride]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def parse_grid(spec):
    """'30' -> (30, 30); '100x50' -> (100 columns, 50 rows)."""
    parts = spec.lower().split("x")
    cols, rows = (int(parts[0]),) * 2 if len(parts) == 1 else map(int, parts)
    return cols, rows


def grid_points(cols, rows, H, W):
    """(rows*cols, 2) row-major (x, y) grid, same layout as CoTracker's get_points_on_a_grid:
    built at model resolution (384x512) with a W/64 margin, then rescaled to (H, W)."""
    mh, mw = 384, 512
    m = mw / 64
    ys = torch.linspace(m, mh - m, rows) if rows > 1 else torch.tensor([mh / 2])
    xs = torch.linspace(m, mw - m, cols) if cols > 1 else torch.tensor([mw / 2])
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    return xy * torch.tensor([(W - 1) / (mw - 1), (H - 1) / (mh - 1)])


def build_queries(args, T, H, W):
    """All query points as a (N, 3) CPU tensor of (t, x, y) in original pixel coords."""
    if args.queries:
        q = np.loadtxt(args.queries, dtype=np.float32).reshape(-1, 3)
        return torch.from_numpy(q)
    xy = grid_points(*parse_grid(args.grid_size), H, W)
    if args.mask:
        m = np.array(Image.open(args.mask).convert("L").resize((W, H), Image.NEAREST)) > 0
        ix = xy[:, 0].round().long().clamp(0, W - 1)
        iy = xy[:, 1].round().long().clamp(0, H - 1)
        xy = xy[torch.from_numpy(m)[iy, ix]]
    t = torch.full((len(xy), 1), float(args.grid_query_frame))
    return torch.cat([t, xy], dim=1)


def run_offline(model, video, queries, args):
    return model(video, queries=queries, backward_tracking=args.backward_tracking)


def run_online(model, video, queries, args):
    # Same loop as the README: overlapping windows of 2*step frames, advancing by step.
    # The last (shorter) chunk is padded inside the model, so the tail is covered.
    T = video.shape[1]
    step = model.step
    model(video_chunk=video, is_first_step=True, queries=queries)
    tracks = vis = None
    for ind in range(0, max(T - step, 1), step):  # at least one window, even for short clips
        tracks, vis = model(video_chunk=video[:, ind : ind + step * 2])
    return tracks[:, :T], vis[:, :T]


def gpu_worker(device, frames, items, ckpt, args):
    """Track work items (segment, chunk of queries) on one device.

    items: [(seg_idx, chunk_idx, start, end, queries)] -> [(seg_idx, chunk_idx, tracks, vis)],
    tracks (end-start, n, 2) for frames [start, end) only."""
    if args.mode == "offline":
        model = CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=60).to(device)
        run = run_offline
    else:
        model = CoTrackerOnlinePredictor(checkpoint=ckpt, window_len=16).to(device)
        run = run_online
    video = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2)[None].float()  # B T C H W
    results = []
    for seg, idx, start, end, q in items:
        if end - start < 2:  # a 1-frame segment can't be tracked: points stay at their queries
            tracks = q[None, None, :, 1:].expand(1, end - start, -1, -1)
            vis = torch.ones(1, end - start, len(q), dtype=torch.bool)
        else:
            tracks, vis = run(model, video[:, start:end], q[None].to(device), args)
        results.append((seg, idx, tracks[0].cpu(), vis[0].cpu()))
        print(f"  [{device}] frames {start}-{end - 1}, chunk {idx}: {len(q)} points done", flush=True)
    del model, video
    torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--mode", choices=["offline", "online"], default="offline")
    p.add_argument("--checkpoint", default=None, help="defaults to checkpoints/scaled_<mode>.pth")
    p.add_argument(
        "--grid_size", default="20", help='"N" for an NxN grid, or "COLSxROWS" e.g. 100x50'
    )
    p.add_argument("--grid_query_frame", type=int, default=0)
    p.add_argument("--queries", default=None, help='text file of "t x y" rows; overrides grid')
    p.add_argument("--mask", default=None, help="binary mask image; keeps grid points inside it")
    p.add_argument("--backward_tracking", action="store_true", help="offline only")
    p.add_argument(
        "--segment_len", type=int, default=10,
        help="videos longer than this are split into non-overlapping segments of this many frames; "
        "each segment starts a fresh grid on its first frame. 0 = track the whole video at once.",
    )
    p.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--gpus", default="all", help='"all" or comma list of visible GPU ids, e.g. 0,1')
    p.add_argument(
        "--chunk_size", type=int, default=0,
        help="max points per forward pass (0 = split points evenly over GPUs). Lower it on OOM.",
    )
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--save_dir", default=None, help="write outputs exactly here (overrides --out_dir)")
    p.add_argument("--fps", type=float, default=None, help="visualization fps (default: source fps / stride)")
    p.add_argument("--radius", type=int, default=4, help="visualization dot radius in px (0 = 1 pixel)")
    p.add_argument("--no_vis", action="store_true")
    args = p.parse_args()

    if args.mode == "online" and args.backward_tracking:
        print("warning: --backward_tracking is ignored in online mode")
    ckpt = args.checkpoint or f"./checkpoints/scaled_{args.mode}.pth"
    if not args.checkpoint and not os.path.isfile(ckpt):
        # The default CoTracker3 weights are not in git: fetch them once (README: "checkpoints" section).
        url = f"https://huggingface.co/facebook/cotracker3/resolve/main/scaled_{args.mode}.pth"
        print(f"downloading {url} -> {ckpt}", flush=True)
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.hub.download_url_to_file(url, ckpt)
    if not torch.cuda.is_available():
        devices = ["cpu"]
    elif args.gpus == "all":
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = [f"cuda:{i}" for i in args.gpus.split(",")]

    frames = load_video(args.video, args.max_frames, args.frame_stride)
    T, H, W, _ = frames.shape
    queries = build_queries(args, T, H, W)
    N = len(queries)
    seg_len = args.segment_len if (args.segment_len > 0 and T > args.segment_len) else 0
    if seg_len and args.queries:
        print("note: --queries are tied to specific frames; tracking the whole video (no segments)")
        seg_len = 0
    if seg_len:
        if args.grid_query_frame:
            print("note: --grid_query_frame is ignored with segments; each segment starts on its first frame")
        queries[:, 0] = 0  # query time is relative to each segment's first frame
        segments = [(s, min(s + seg_len, T)) for s in range(0, T, seg_len)]
    else:
        segments = [(0, T)]
    chunk_size = args.chunk_size or -(-N // len(devices))
    chunks = list(queries.split(chunk_size))
    print(
        f"video: {T} frames, {W}x{H}, mode={args.mode}, {N} points in "
        f"{len(chunks)} chunk(s) of <= {chunk_size} on {devices}, "
        + (f"{len(segments)} segments of {seg_len} frames" if seg_len else "whole video at once")
    )

    # Every (segment, chunk) pair is one work item; round-robin over devices, one thread per
    # device (CUDA work releases the GIL).
    items = [(si, ci, s, e, q) for si, (s, e) in enumerate(segments) for ci, q in enumerate(chunks)]
    per_device = [items[i :: len(devices)] for i in range(len(devices))]
    with ThreadPoolExecutor(len(devices)) as ex:
        futures = [
            ex.submit(gpu_worker, d, frames, it, ckpt, args)
            for d, it in zip(devices, per_device) if it
        ]
        results = sorted((r for f in futures for r in f.result()), key=lambda r: (r[0], r[1]))
    # Chunks side by side within a segment, segments one after another in time.
    tracks = torch.cat([
        torch.cat([r[2] for r in results if r[0] == si], dim=1) for si in range(len(segments))
    ], dim=0)  # (T, N, 2)
    vis = torch.cat([
        torch.cat([r[3] for r in results if r[0] == si], dim=1) for si in range(len(segments))
    ], dim=0)  # (T, N)

    name = os.path.splitext(os.path.basename(args.video))[0]
    out = args.save_dir or os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "tracks.npy"), tracks.numpy().astype(np.float32))
    np.save(os.path.join(out, "visibility.npy"), vis.numpy().astype(bool))
    np.save(os.path.join(out, "queries.npy"), queries.numpy())
    meta = dict(video=args.video, width=W, height=H, frames=T, frame_stride=args.frame_stride,
                fps=iio.immeta(args.video, plugin="FFMPEG").get("fps"), segment_len=seg_len)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"tracks {tuple(tracks.shape)}, visible fraction {vis.float().mean():.2f} -> {out}")

    if not args.no_vis:
        query_frame = 0 if (args.backward_tracking or seg_len) else args.grid_query_frame
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
        # show_first_frame=0 and our own writer: Visualizer.save_video otherwise repeats the
        # first frame and drops frames ([2:-1]), so tracks.mp4 would not line up with the source.
        res = Visualizer(pad_value=0, point_radius=args.radius, show_first_frame=0).visualize(
            video, tracks[None], vis[None], query_frame=query_frame, save_video=False
        )  # (1, T, 3, H, W) uint8
        fps = args.fps or (meta["fps"] or 10 * args.frame_stride) / args.frame_stride
        iio.imwrite(os.path.join(out, "tracks.mp4"), res[0].permute(0, 2, 3, 1).numpy(),
                    fps=fps, plugin="FFMPEG")
        print(f"saved {os.path.join(out, 'tracks.mp4')} ({T} frames, same as source)")


if __name__ == "__main__":
    main()
