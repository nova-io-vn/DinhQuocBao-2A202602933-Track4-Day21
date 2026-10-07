"""Topic D: classical LiDAR obstacle detection and parameter stress test.

Pipeline: ROI/finite filtering -> voxel downsampling -> RANSAC ground removal
-> DBSCAN clustering -> axis-aligned 3-D bounding boxes.  The script writes all
CSV and figures used by ``report/REPORT.md`` and is intentionally headless so it
can be reproduced on a CPU-only machine.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from starter.datasets import load_points


@dataclass
class Detection:
    label: int
    n_points: int
    center: np.ndarray
    extent: np.ndarray
    min_bound: np.ndarray
    max_bound: np.ndarray
    distance_m: float


def finite_roi(points: np.ndarray, x_range=(0.0, 40.0), y_range=(-20.0, 20.0),
               z_range=(-3.0, 3.0)) -> np.ndarray:
    """Remove invalid points and retain the robot's forward region of interest."""
    p = points[np.isfinite(points[:, :3]).all(axis=1)]
    keep = (
        (p[:, 0] >= x_range[0]) & (p[:, 0] <= x_range[1])
        & (p[:, 1] >= y_range[0]) & (p[:, 1] <= y_range[1])
        & (p[:, 2] >= z_range[0]) & (p[:, 2] <= z_range[1])
    )
    return p[keep]


def to_cloud(points_xyz: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64, copy=False))
    return cloud


def cluster_boxes(points_xyz: np.ndarray, labels: np.ndarray, min_cluster_points: int = 6,
                  max_extent=(12.0, 8.0, 4.0)) -> list[Detection]:
    detections: list[Detection] = []
    for label in sorted(set(labels.tolist()) - {-1}):
        cluster = points_xyz[labels == label]
        if len(cluster) < min_cluster_points:
            continue
        lo, hi = cluster.min(axis=0), cluster.max(axis=0)
        extent = hi - lo
        # Reject huge residual road/wall components that are not local obstacles.
        if np.any(extent > np.asarray(max_extent)) or extent[2] < 0.05:
            continue
        detections.append(Detection(
            label=int(label), n_points=len(cluster), center=(lo + hi) / 2,
            extent=extent, min_bound=lo, max_bound=hi,
            distance_m=float(np.linalg.norm(cluster[:, :2], axis=1).min()),
        ))
    return detections


def run_pipeline(points: np.ndarray, voxel_size: float, ground_threshold: float,
                 eps: float, min_points: int, seed: int) -> dict:
    """Run one deterministic configuration and return points, labels and metrics."""
    o3d.utility.random.seed(seed)
    roi = finite_roi(points)
    down = to_cloud(roi[:, :3]).voxel_down_sample(voxel_size)
    xyz_down = np.asarray(down.points)
    plane, ground_idx = down.segment_plane(
        distance_threshold=ground_threshold, ransac_n=3, num_iterations=150)
    obstacle = down.select_by_index(ground_idx, invert=True)
    xyz_obstacle = np.asarray(obstacle.points)
    labels = np.asarray(obstacle.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=False), dtype=np.int32)
    detections = cluster_boxes(xyz_obstacle, labels, min_cluster_points=min_points)
    return {
        "roi": roi[:, :3], "down": xyz_down, "ground_idx": np.asarray(ground_idx),
        "obstacle": xyz_obstacle, "labels": labels, "plane": np.asarray(plane),
        "detections": detections,
    }


def _draw_boxes(ax, detections: list[Detection], color="black") -> None:
    for d in detections:
        lo, hi = d.min_bound, d.max_bound
        xs = [lo[0], hi[0], hi[0], lo[0], lo[0]]
        ys = [lo[1], lo[1], hi[1], hi[1], lo[1]]
        ax.plot(xs, ys, color=color, linewidth=1.0)


def _bev(ax, xyz: np.ndarray, title: str, color=None, size=0.35) -> None:
    if len(xyz):
        ax.scatter(xyz[:, 0], xyz[:, 1], c=color if color is not None else xyz[:, 2],
                   s=size, cmap="turbo", rasterized=True)
    ax.set(xlim=(0, 40), ylim=(-20, 20), aspect="equal", xlabel="x forward (m)", ylabel="y left (m)")
    ax.set_title(title)
    ax.grid(alpha=0.15)


def save_demo(result: dict, path: Path, voxel: float, threshold: float, eps: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    _bev(axes[0, 0], result["roi"], f"1. ROI input: {len(result['roi']):,} points")
    _bev(axes[0, 1], result["down"], f"2. Voxel {voxel:.2f} m: {len(result['down']):,} points")
    _bev(axes[1, 0], result["obstacle"],
         f"3. Remove ground (threshold={threshold:.2f} m): {len(result['obstacle']):,} left")
    labels = result["labels"]
    colors = np.where(labels >= 0, labels, np.nan) if len(labels) else None
    _bev(axes[1, 1], result["obstacle"],
         f"4. DBSCAN eps={eps:.2f}: {len(result['detections'])} obstacles", colors, size=1.0)
    _draw_boxes(axes[1, 1], result["detections"])
    fig.suptitle("Classical LiDAR obstacle pipeline — KITTI frame 000011", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_occupancy(result: dict, path: Path, resolution=0.20) -> None:
    xyz = result["obstacle"]
    x_edges = np.arange(0, 40 + resolution, resolution)
    y_edges = np.arange(-20, 20 + resolution, resolution)
    grid, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=(x_edges, y_edges))
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.imshow(grid.T > 0, origin="lower", extent=(0, 40, -20, 20), cmap="binary", aspect="equal")
    _draw_boxes(ax, result["detections"], color="tab:red")
    ax.plot(0, 0, marker="^", color="tab:blue", markersize=10, label="LiDAR")
    ax.set(title=f"BEV occupancy grid ({resolution:.2f} m/cell)", xlabel="x forward (m)", ylabel="y left (m)")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def benchmark(points: np.ndarray, voxels: list[float], thresholds: list[float], eps: float,
              min_points: int, repeats: int, seed: int) -> list[dict]:
    rows = []
    for voxel in voxels:
        for threshold in thresholds:
            # Warm-up is deliberately excluded from the latency distribution.
            run_pipeline(points, voxel, threshold, eps, min_points, seed)
            times, final = [], None
            for _ in range(repeats):
                start = time.perf_counter()
                final = run_pipeline(points, voxel, threshold, eps, min_points, seed)
                times.append((time.perf_counter() - start) * 1000)
            assert final is not None
            dets = final["detections"]
            rows.append({
                "voxel_size_m": voxel, "ground_threshold_m": threshold,
                "eps_m": eps, "min_points": min_points, "seed": seed,
                "n_input_roi": len(final["roi"]), "n_downsampled": len(final["down"]),
                "n_ground": len(final["ground_idx"]), "n_obstacle_points": len(final["obstacle"]),
                "n_clusters": len(dets),
                "nearest_obstacle_m": min((d.distance_m for d in dets), default=np.nan),
                "median_cluster_points": float(np.median([d.n_points for d in dets])) if dets else np.nan,
                "latency_p50_ms": float(np.percentile(times, 50)),
                "latency_p95_ms": float(np.percentile(times, 95)),
                "latency_repeats": repeats,
            })
            print(f"voxel={voxel:.2f} ground={threshold:.2f}: clusters={len(dets):2d}, "
                  f"nearest={rows[-1]['nearest_obstacle_m']:.2f} m, p50={rows[-1]['latency_p50_ms']:.1f} ms")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_benchmark(rows: list[dict], path: Path) -> None:
    voxels = sorted({r["voxel_size_m"] for r in rows})
    thresholds = sorted({r["ground_threshold_m"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, key, title, fmt in zip(
        axes,
        ("n_clusters", "nearest_obstacle_m", "latency_p50_ms"),
        ("Detected clusters", "Nearest obstacle (m)", "Latency p50 (ms)"),
        (".0f", ".2f", ".1f"),
    ):
        mat = np.array([[next(r[key] for r in rows if r["voxel_size_m"] == v and
                              r["ground_threshold_m"] == t) for t in thresholds] for v in voxels])
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        for i in range(len(voxels)):
            for j in range(len(thresholds)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color="white", fontsize=9)
        ax.set(xticks=range(len(thresholds)), xticklabels=thresholds,
               yticks=range(len(voxels)), yticklabels=voxels,
               xlabel="ground threshold (m)", ylabel="voxel size (m)", title=title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_failure(points: np.ndarray, path: Path, voxel: float, eps: float,
                 min_points: int, seed: int, good_threshold=0.10, bad_threshold=0.40) -> dict:
    good = run_pipeline(points, voxel, good_threshold, eps, min_points, seed)
    bad = run_pipeline(points, voxel, bad_threshold, eps, min_points, seed)
    a, b, c, d = good["plane"]
    signed_height = (good["down"] @ np.array([a, b, c]) + d) / np.linalg.norm([a, b, c])
    low = (signed_height > good_threshold) & (signed_height <= bad_threshold)
    good_set = {tuple(np.round(p, 5)) for p in good["obstacle"]}
    bad_set = {tuple(np.round(p, 5)) for p in bad["obstacle"]}
    low_points = good["down"][low]
    retained_good = sum(tuple(np.round(p, 5)) in good_set for p in low_points)
    retained_bad = sum(tuple(np.round(p, 5)) in bad_set for p in low_points)
    lost_ratio = 1.0 - retained_bad / max(retained_good, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, result, threshold in zip(axes, (good, bad), (good_threshold, bad_threshold)):
        _bev(ax, result["obstacle"],
             f"threshold={threshold:.2f} m\n{len(result['obstacle']):,} obstacle points, "
             f"{len(result['detections'])} clusters", size=0.8)
        if len(low_points):
            ax.scatter(low_points[:, 0], low_points[:, 1], s=4, facecolors="none", edgecolors="red",
                       linewidths=0.35, label=f"low returns ({good_threshold:.2f}–{bad_threshold:.2f} m)")
        _draw_boxes(ax, result["detections"])
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Failure: aggressive ground threshold removes {lost_ratio:.1%} of low returns", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"low_points_reference": retained_good, "low_points_bad": retained_bad,
            "low_points_lost_ratio": lost_ratio,
            "clusters_good": len(good["detections"]), "clusters_bad": len(bad["detections"])}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Topic D: CPU LiDAR obstacle detection benchmark")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", default="000011")
    ap.add_argument("--voxel-sizes", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    ap.add_argument("--ground-thresholds", nargs="+", type=float, default=[0.10, 0.20, 0.40])
    ap.add_argument("--eps", type=float, default=0.65)
    ap.add_argument("--min-points", type=int, default=6)
    ap.add_argument("--latency-repeats", type=int, default=21)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.latency_repeats < 20:
        print("Warning: rubric asks for >=20 timed repeats; use --latency-repeats 21 for final evidence.")
    points = load_points(args.data_root, args.frame)
    out = Path(args.out_dir)
    rows = benchmark(points, args.voxel_sizes, args.ground_thresholds, args.eps,
                     args.min_points, args.latency_repeats, args.seed)
    write_csv(rows, out / "obstacle_sweep.csv")
    save_benchmark(rows, out / "figures" / "obstacle_benchmark.png")

    baseline_voxel, baseline_threshold = 0.20, 0.10
    baseline = run_pipeline(points, baseline_voxel, baseline_threshold, args.eps, args.min_points, args.seed)
    save_demo(baseline, out / "figures" / "obstacle_pipeline_demo.png",
              baseline_voxel, baseline_threshold, args.eps)
    save_occupancy(baseline, out / "figures" / "occupancy_grid.png")
    failure = save_failure(points, out / "figures" / "fail_01_low_obstacle_removed.png",
                           baseline_voxel, args.eps, args.min_points, args.seed)
    write_csv([failure], out / "failure_metrics.csv")
    print(f"Wrote benchmark and figures under {out}")


if __name__ == "__main__":
    main()
