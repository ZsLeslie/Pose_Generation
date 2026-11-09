"""Colored PLY output and optional filters adapted from the supplied laser_ply.py."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def voxel_downsample(points, colors, voxel_size):
    if not len(points) or voxel_size <= 0:
        return points, colors
    # Unique integer triplets avoid collisions from packed signed voxel indices.
    indices = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    centers = np.zeros((len(counts), 3), dtype=np.float64)
    averaged_colors = np.zeros_like(centers)
    np.add.at(centers, inverse, points)
    np.add.at(averaged_colors, inverse, colors)
    return centers / counts[:, None], np.clip(averaged_colors / counts[:, None], 0, 255).astype(np.uint8)


def statistical_outlier_filter(points, colors, neighbors=30, std_ratio=3.0):
    if neighbors <= 0 or len(points) < neighbors + 2:
        return points, colors
    distances, _ = cKDTree(points).query(points, k=neighbors + 1, workers=-1)
    means = distances[:, 1:].mean(axis=1)
    keep = means <= means.mean() + std_ratio * means.std()
    return points[keep], colors[keep]


def radius_outlier_filter(points, colors, radius=0.2, min_neighbors=2):
    if radius <= 0 or not len(points):
        return points, colors
    # Match the source script: the point itself counts as a radius neighbor.
    counts = cKDTree(points).query_ball_point(points, r=radius, return_length=True, workers=-1)
    keep = counts >= min_neighbors
    return points[keep], colors[keep]


def write_ply(points, colors, path):
    """Write the source script's XYZ + grayscale/RGB ASCII PLY format."""
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("PLY requires equally sized N x 3 position and color arrays")
    if not np.isfinite(points).all():
        raise ValueError("PLY contains nonfinite coordinates")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for start in range(0, len(points), 65536):
            values = np.column_stack([points[start:start + 65536], colors[start:start + 65536]])
            np.savetxt(stream, values, fmt=["%.6f"] * 3 + ["%d"] * 3)
