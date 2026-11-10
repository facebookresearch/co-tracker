"""Compare two tracking results (from scripts/track_video.py) and draw the difference on video A.

Points are paired by their query position (same normalized (x, y) at the same query frame), so
both runs should use the same grid (e.g. both --grid_size 50). Resolutions may differ: B is
rescaled to A's pixel space.

Metrics (--metric), vector = A - B per point and frame:
    position (default)      A[t] - B[t]: difference of absolute positions
    displacement            (A[t] - A[q]) - (B[t] - B[q]): difference of movement since the
                            query frame q (since the segment start for segmented tracking). Identical to "position" when both runs start every
                            point at the same pixel (same grid, same resolution).

Example:
    python scripts/compare_tracks.py --a outputs/run_A --b outputs/run_B --video A.mp4

Outputs (in --out, default outputs/compare/<A>_vs_<B>/):
    diff_vec.npy    float32 (T, N, 2) vector A - B (dx, dy) in A pixels, NaN where either is occluded
    diff.npy        float32 (T, N)  its length |A - B|
    b_on_a.npy      float32 (T, N, 2) where B's point lands in A's pixel space
    pairs.npy       int     (N, 2)  matched point indices (index in A, index in B)
    per_frame.csv   frame, mean, median, p95, max, n_valid
    summary.json    overall mean / median over all compared points and frames, and the area
                    under the per-frame mean and median curves (px * frames)
    diff_over_time.png
    paired.npy      bool    (T, N_A) A point has a B partner that is visible in both at frame t
                                     (and inside the mask, with --mask)
    in_mask.npy     bool    (T, N)  with --mask: pair is inside the mask at frame t
    compare.mp4     video A. --view pairs (default): green = paired at that frame, gray = no
                    pair (no B partner, or hidden in A or B). --view diff: colored by |A - B|.
"""

import argparse
import csv
import json
import os

import cv2
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(d, query_frame):
    tracks = np.load(os.path.join(d, "tracks.npy"))  # (T, N, 2)
    vis = np.load(os.path.join(d, "visibility.npy"))  # (T, N)
    qpath = os.path.join(d, "queries.npy")
    if os.path.exists(qpath):
        queries = np.load(qpath)
    else:  # older runs: no queries saved; tracks equal the query point on the query frame
        print(f"note: {d} has no queries.npy, assuming all points start on frame {query_frame}")
        queries = np.concatenate(
            [np.full((tracks.shape[1], 1), query_frame, np.float32), tracks[query_frame]], 1
        )
    mpath = os.path.join(d, "meta.json")
    meta = json.load(open(mpath)) if os.path.exists(mpath) else {}
    return tracks, vis, queries, meta


def size_of(meta, override, fallback_video):
    if override:
        w, h = map(int, override.lower().split("x"))
        return w, h
    if "width" in meta:
        return meta["width"], meta["height"]
    if fallback_video:
        h, w = iio.imread(fallback_video, index=0, plugin="FFMPEG").shape[:2]
        return w, h
    return None


def match_points(qa, qb, size_a, size_b, tol):
    """Pair points with the same query frame and normalized query (x, y)."""
    na = qa[:, 1:] / (np.array(size_a) - 1)
    nb = qb[:, 1:] / (np.array(size_b) - 1)
    if len(qa) == len(qb) and np.allclose(na, nb, atol=tol) and (qa[:, 0] == qb[:, 0]).all():
        idx = np.arange(len(qa))
        return np.stack([idx, idx], 1)
    pairs = []
    for t in np.unique(qa[:, 0]):
        ia, ib = np.where(qa[:, 0] == t)[0], np.where(qb[:, 0] == t)[0]
        if len(ib) == 0:
            continue
        d = np.linalg.norm(na[ia, None] - nb[None, ib], axis=-1)
        j = d.argmin(1)
        ok = d[np.arange(len(ia)), j] < tol
        pairs.append(np.stack([ia[ok], ib[j[ok]]], 1))
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), int)


def align_time(tracks, vis, T):
    """Linearly resample (Tb, N, ...) to T frames over the same duration."""
    Tb = len(tracks)
    s = np.linspace(0, Tb - 1, T)
    i0 = np.floor(s).astype(int)
    i1 = np.minimum(i0 + 1, Tb - 1)
    w = (s - i0)[:, None, None]
    return tracks[i0] * (1 - w) + tracks[i1] * w, vis[np.round(s).astype(int)]


def colorbar(frame, vmax, cmap_lut):
    H, W = frame.shape[:2]
    bh, bw, x0, y0 = int(H * 0.4), 14, W - 60, 20
    for k in range(bh):
        c = cmap_lut[int(255 * (1 - k / (bh - 1)))]
        cv2.line(frame, (x0, y0 + k), (x0 + bw, y0 + k), c, 1)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), 1)
    for txt, y in [(f"{vmax:.1f}", y0 + 5), ("0", y0 + bh)]:
        cv2.putText(frame, txt, (x0 + bw + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(frame, txt, (x0 + bw + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "px", (x0, y0 + bh + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def mask_selection(args, A, B, T, size_a, stride):
    """Which pairs lie inside the mask, consulting it every args.mask_step frames.

    A, B: (T, N, 2) positions in A's pixel space. Returns in_mask (T, N) bool and
    {frame: (H, W) bool mask} for the consulted frames (for drawing)."""
    W, H = size_a
    # (T, H, W) bool per source frame, e.g. X_demo_008_01_mask_stack.npy; or one (H, W) mask.
    stack = np.load(args.mask, mmap_mode="r")  # memory-mapped: only consulted frames are read
    if stack.ndim == 2:
        stack = stack[None]
    if stack.ndim != 3:
        raise SystemExit(f"mask {args.mask}: expected (T, H, W) or (H, W), got {stack.shape}")
    if len(stack) > 1:
        stack = stack[::stride]  # mask frames follow the source video
    Tm, Hm, Wm = stack.shape
    if (Hm, Wm) != (H, W):
        print(f"note: mask is {Wm}x{Hm}, sources are {W}x{H}; resizing mask")
    if 1 < Tm < T:
        print(f"note: mask has {Tm} frames, tracks have {T}; reusing its last frame")
    step = max(1, args.mask_step)
    in_mask = np.zeros(A.shape[:2], bool)
    masks = {}

    def inside(P, m):
        x, y = np.round(P[:, 0]).astype(int), np.round(P[:, 1]).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        res = np.zeros(len(P), bool)
        res[ok] = m[y[ok], x[ok]]
        return res

    for m0 in range(0, T, step):
        m = np.asarray(stack[min(m0, Tm - 1)]) > 0  # True / non-zero = inside
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        masks[m0] = m
        ia, ib = inside(A[m0], m), inside(B[m0], m)
        sel = ia & ib if args.mask_rule == "both" else ia if args.mask_rule == "a" else ia | ib
        in_mask[m0 : m0 + step] = sel
    print(f"mask: consulted every {step} frames ({len(masks)} times), rule '{args.mask_rule}'; "
          f"on average {in_mask.sum(1).mean():.0f} of {A.shape[1]} points inside")
    return in_mask, masks


def curve_auc(y):
    """Area under a per-frame curve (trapezoid rule, 1 frame apart), in px * frames.
    Frames without compared points (NaN) break the curve; those gaps add nothing."""
    y = np.asarray(y, float)
    both = ~np.isnan(y[:-1]) & ~np.isnan(y[1:])
    return float(((y[:-1] + y[1:]) / 2)[both].sum())


def summarize(diff, stats):
    """diff: (T, N) per-point differences (NaN = not compared); stats: per-frame rows of
    [mean, median, p95, max]."""
    has = ~np.isnan(diff)
    num = lambda v: None if v is None or np.isnan(v) else round(float(v), 4)
    return dict(
        overall_mean=num(diff[has].mean()) if has.any() else None,
        overall_median=num(np.median(diff[has])) if has.any() else None,
        auc_mean=num(curve_auc(stats[:, 0])),
        auc_median=num(curve_auc(stats[:, 1])),
        auc_unit="px*frames",
        frames=int(len(diff)),
        frames_compared=int((~np.isnan(stats[:, 0])).sum()),
        points_compared=int(has.sum()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="track output dir of video A (drawn on)")
    p.add_argument("--b", required=True, help="track output dir of video B")
    p.add_argument("--video", default=None, help="video A; defaults to the path in A's meta.json")
    p.add_argument("--frame_stride", type=int, default=None, help="as used for A's tracking")
    p.add_argument("--size_a", default=None, help="WxH of video A if A has no meta.json")
    p.add_argument("--size_b", default=None, help="WxH of video B if B has no meta.json")
    p.add_argument("--query_frame", type=int, default=0, help="for runs without queries.npy")
    p.add_argument("--metric", choices=["position", "displacement"], default="position")
    p.add_argument(
        "--time", choices=["truncate", "resample"], default="truncate",
        help="different frame counts: cut to the shorter one, or stretch B to A's length",
    )
    p.add_argument("--match_tol", type=float, default=2e-3, help="normalized query distance")
    p.add_argument("--vmax", type=float, default=None, help="colormap max in px (default: p95)")
    p.add_argument(
        "--view", choices=["pairs", "diff"], default="pairs",
        help="pairs: highlight A points that have a B pair; diff: color by |A - B|",
    )
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--fps", type=float, default=None, help="default: source fps / stride")
    p.add_argument("--out", default=None)
    p.add_argument("--mask", default=None,
                   help=".npy mask, (T, H, W) bool per source frame (or one (H, W)), same size as "
                   "the sources; only points inside its True area are compared")
    p.add_argument("--mask_step", type=int, default=5,
                   help="consult the mask every N frames (frames 0, N, 2N, ...); the selection "
                   "holds until the next one")
    p.add_argument("--mask_rule", choices=["both", "a", "either"], default="both",
                   help="point counts if inside the mask in both A and B / in A only / in either")
    args = p.parse_args()

    A, visA, qA, metaA = load_run(args.a, args.query_frame)
    B, visB, qB, metaB = load_run(args.b, args.query_frame)
    video_path = args.video or metaA.get("video")
    if video_path is None:
        raise SystemExit("A has no meta.json; pass --video (the video A was tracked on)")
    if "video" in metaA and os.path.realpath(video_path) != os.path.realpath(metaA["video"]):
        print(f"WARNING: --video {video_path} is not the video A was tracked on "
              f"({metaA['video']}); A's points will be drawn on the wrong footage")
    elif "video" not in metaA:
        print(f"note: drawing A's tracks ({args.a}) on {video_path}; make sure that is A's video")
    size_a = size_of(metaA, args.size_a, video_path)
    size_b = size_of(metaB, args.size_b, metaB.get("video"))
    if size_b is None:
        print(f"note: B size unknown (no meta.json / --size_b), assuming same as A {size_a}")
        size_b = size_a

    # Pair points and bring B into A's pixel space.
    pairs = match_points(qA, qB, size_a, size_b, args.match_tol)
    print(f"matched {len(pairs)} points (A has {A.shape[1]}, B has {B.shape[1]})")
    if len(pairs) == 0:
        raise SystemExit("no matching query points; track both videos with the same grid")
    scale = (np.array(size_a) - 1) / (np.array(size_b) - 1)
    A_all, visA_all = A, visA  # every A point, for drawing unpaired ones too
    A, visA, qA = A[:, pairs[:, 0]], visA[:, pairs[:, 0]], qA[pairs[:, 0]]
    B, visB = B[:, pairs[:, 1]] * scale, visB[:, pairs[:, 1]]

    if len(A) != len(B):
        if args.time == "resample":
            print(f"frames differ (A {len(A)}, B {len(B)}): resampling B to {len(A)} frames")
            B, visB = align_time(B, visB, len(A))
        else:
            T = min(len(A), len(B))
            print(f"frames differ (A {len(A)}, B {len(B)}): truncating to {T} (see --time)")
            A, visA, B, visB = A[:T], visA[:T], B[:T], visB[:T]
    T, N = A.shape[:2]
    A_all, visA_all = A_all[:T], visA_all[:T]

    seg_a, seg_b = metaA.get("segment_len") or 0, metaB.get("segment_len") or 0
    if seg_a != seg_b:
        print(f"WARNING: A and B were tracked with different segment lengths ({seg_a} vs {seg_b}); "
              "their points restart on different frames")
    if args.metric == "displacement" and seg_a:
        # Segmented tracking: movement is measured from the start of each point's current segment.
        ref = (np.arange(T) // seg_a) * seg_a
        b_on_a = A[ref] + (B - B[ref])
    elif args.metric == "displacement":
        tq = np.clip(qA[:, 0].astype(int), 0, T - 1)
        ar = np.arange(N)
        b_on_a = A[tq, ar][None] + (B - B[tq, ar][None])
    else:
        b_on_a = B
    valid = visA & visB
    stride = args.frame_stride or metaA.get("frame_stride", 1)
    mask_frames = None
    if args.mask:
        in_mask, mask_frames = mask_selection(args, A, B, T, size_a, stride)
        valid &= in_mask
    diff_vec = A - b_on_a  # (T, N, 2)
    diff_vec[~valid] = np.nan
    diff = np.linalg.norm(diff_vec, axis=-1)
    paired = np.zeros(visA_all.shape, bool)  # (T, N_A)
    paired[:, pairs[:, 0]] = valid

    name = lambda d: os.path.basename(os.path.normpath(d))
    out = args.out or os.path.join("outputs", "compare", f"{name(args.a)}_vs_{name(args.b)}")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "diff_vec.npy"), diff_vec.astype(np.float32))
    np.save(os.path.join(out, "diff.npy"), diff.astype(np.float32))
    np.save(os.path.join(out, "b_on_a.npy"), b_on_a.astype(np.float32))
    np.save(os.path.join(out, "pairs.npy"), pairs)
    np.save(os.path.join(out, "paired.npy"), paired)
    if mask_frames is not None:
        np.save(os.path.join(out, "in_mask.npy"), in_mask)

    stats = []
    with open(os.path.join(out, "per_frame.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "mean", "median", "p95", "max", "n_valid"])
        for t in range(T):
            d = diff[t][~np.isnan(diff[t])]
            row = [np.mean(d), np.median(d), np.percentile(d, 95), np.max(d)] if len(d) else [np.nan] * 4
            stats.append(row)
            w.writerow([t] + [f"{x:.3f}" for x in row] + [len(d)])
    stats = np.array(stats)
    summary = summarize(diff, stats)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary: AUC mean {summary['auc_mean']} / median {summary['auc_median']} px*frames, "
          f"overall mean {summary['overall_mean']} px, overall median {summary['overall_median']} px")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(stats[:, 0], label="mean")
    ax.plot(stats[:, 1], label="median")
    ax.plot(stats[:, 2], label="95th pct", linestyle="--")
    ax.set_xlabel("frame")
    ax.set_ylabel(f"{args.metric} difference (px, A space)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "diff_over_time.png"), dpi=120)

    vmax = args.vmax or float(np.nanpercentile(diff, 95)) or 1.0
    print(f"overall: mean {np.nanmean(diff):.2f} px, median {np.nanmedian(diff):.2f} px, "
          f"p95 {np.nanpercentile(diff, 95):.2f} px; colormap max {vmax:.2f} px")

    # Draw on video A (same frame subsampling as A's tracking).
    frames = iio.imread(video_path, plugin="FFMPEG")[::stride][:T]
    if len(frames) < T:
        raise SystemExit(f"video A has {len(frames)} frames after stride, tracks have {T}")
    lut = (matplotlib.colormaps["turbo"](np.linspace(0, 1, 256))[:, :3] * 255).astype(int)
    lut = [tuple(int(v) for v in c) for c in lut]
    src_fps = metaA.get("fps") or iio.immeta(video_path, plugin="FFMPEG").get("fps") or 10 * stride
    fps = args.fps or src_fps / stride

    out_frames = []
    green, gray = (0, 255, 0), (160, 160, 160)
    for t in range(T):
        fr = np.ascontiguousarray(frames[t])
        if mask_frames is not None:  # outline of the mask consulted for this frame
            cs, _ = cv2.findContours(mask_frames[(t // args.mask_step) * args.mask_step].astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(fr, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
        if args.view == "pairs":
            for i in np.where(~paired[t])[0]:
                cv2.circle(fr, tuple(np.round(A_all[t, i]).astype(int)), args.radius, gray, -1, cv2.LINE_AA)
            for i in np.where(paired[t])[0]:
                cv2.circle(fr, tuple(np.round(A_all[t, i]).astype(int)), args.radius, green, -1, cv2.LINE_AA)
            label = f"frame {t}  paired {paired[t].sum()}/{paired.shape[1]}"
        else:
            for i in np.where(~valid[t])[0]:  # occluded in A or B: small gray ring
                cv2.circle(fr, tuple(np.round(A[t, i]).astype(int)), args.radius, gray, 1)
            order = np.where(valid[t])[0]
            order = order[np.argsort(diff[t, order])]  # largest differences drawn on top
            for i in order:
                c = lut[int(np.clip(diff[t, i] / vmax, 0, 1) * 255)]
                cv2.circle(fr, tuple(np.round(A[t, i]).astype(int)), args.radius, c, -1, cv2.LINE_AA)
            colorbar(fr, vmax, lut)
            label = f"frame {t}  mean {stats[t, 0]:.2f}px"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(fr, (4, 4), (16 + tw, 14 + th), (0, 0, 0), -1)
        cv2.putText(fr, label, (10, 9 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        out_frames.append(fr)
    iio.imwrite(os.path.join(out, "compare.mp4"), np.stack(out_frames), fps=fps, plugin="FFMPEG")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
