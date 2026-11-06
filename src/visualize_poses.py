"""Plot synchronized front/left/mid/right trajectories from poses.csv."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from generate_poses import PROJECT_ROOT, quaternion_to_rotation


CAMERAS = ("front", "left", "mid", "right")
COLORS = dict(zip(CAMERAS, ("#2563eb", "#ea580c", "#059669", "#9333ea")))
LABELS = {"front": "前相机 front", "left": "左相机 left",
          "mid": "中相机 mid", "right": "右相机 right"}
SYMBOLS = dict(zip(CAMERAS, ("circle", "diamond", "square", "cross")))


@dataclass
class TrajectoryData:
    image_names: list[str]
    centers: dict[str, np.ndarray]
    forward: dict[str, np.ndarray]
    meters_per_colmap_unit: float
    scale_status: str
    front_frame: str

    @property
    def frame_count(self):
        return len(self.image_names)


def natural_key(name):
    return tuple((0, int(part)) if part.isdigit() else (1, part)
                 for part in re.split(r"(\d+)", name))


def load_trajectories(path: Path) -> TrajectoryData:
    """Use camera centers, NOT world-to-camera translation vectors, as positions."""
    required = {"front_image_name", "source_image_id", "source_camera_id", "camera",
                "qw", "qx", "qy", "qz", "tx", "ty", "tz", "cx", "cy", "cz",
                "meters_per_colmap_unit", "scale_status", "front_frame"}
    grouped = {}
    metadata = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                camera = row["camera"]
                if camera not in CAMERAS:
                    raise ValueError(f"Unknown camera: {camera}")
                name = row["front_image_name"]
                if not name:
                    raise ValueError("Empty front image name")
                frame = grouped.setdefault(name, {})
                if camera in frame:
                    raise ValueError(f"Duplicate {camera} pose for {name}")
                rotation = quaternion_to_rotation([float(row[key]) for key in ("qw", "qx", "qy", "qz")])
                translation = np.array([float(row[key]) for key in ("tx", "ty", "tz")])
                center = np.array([float(row[key]) for key in ("cx", "cy", "cz")])
                if not np.isfinite(center).all() or not np.isfinite(translation).all():
                    raise ValueError("Nonfinite position/translation")
                if not np.allclose(center, -rotation.T @ translation, atol=1e-7, rtol=1e-7):
                    raise ValueError("Camera center disagrees with world-to-camera pose")
                scale = float(row["meters_per_colmap_unit"])
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("Scale must be finite and positive")
                metadata.add((scale, row["scale_status"], row["front_frame"]))
                frame[camera] = (center, rotation.T[:, 2],
                                 int(row["source_image_id"]), int(row["source_camera_id"]))
            except (ValueError, TypeError) as error:
                raise ValueError(f"CSV line {line_number}: {error}") from error
    if not grouped:
        raise ValueError("CSV contains no poses")
    if len(metadata) != 1:
        raise ValueError("Mixed scales or front frames in CSV; regenerate a consistent export")
    names = sorted(grouped, key=natural_key)
    for name, frame in grouped.items():
        missing = set(CAMERAS) - frame.keys()
        if missing:
            raise ValueError(f"{name} is missing cameras {sorted(missing)}. "
                             "For older three-camera CSV files, rerun src/generate_poses.py.")
        if len({pose[2:] for pose in frame.values()}) != 1:
            raise ValueError(f"Mismatched source frame IDs for {name}")
    scale, status, front_frame = next(iter(metadata))
    return TrajectoryData(
        names,
        {camera: np.array([grouped[name][camera][0] for name in names]) for camera in CAMERAS},
        {camera: np.array([grouped[name][camera][1] for name in names]) for camera in CAMERAS},
        scale, status, front_frame,
    )


def trajectory_summary(data: TrajectoryData) -> list[dict]:
    result = []
    for camera in CAMERAS:
        points = data.centers[camera]
        result.append({
            "camera": camera, "frame_count": len(points), "length_unit": "colmap_unit",
            "path_length": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
            "start_end_distance": float(np.linalg.norm(points[-1] - points[0])),
            **{f"start_{axis}": float(points[0, i]) for i, axis in enumerate("xyz")},
            **{f"end_{axis}": float(points[-1, i]) for i, axis in enumerate("xyz")},
            "scale_status": data.scale_status,
            "meters_per_colmap_unit": data.meters_per_colmap_unit,
        })
    return result


def scale_caption(data: TrajectoryData, chinese=False) -> str:
    if chinese:
        status = "未定标，位置为暂定值" if data.scale_status == "assumed" else "使用所提供尺度"
        return f"COLMAP 世界坐标 · {data.frame_count} 帧 · {status} · {data.meters_per_colmap_unit:g} m / COLMAP unit"
    status = "UNKNOWN SCALE: provisional positions" if data.scale_status == "assumed" else "Provided scale"
    return f"{data.frame_count} frames | COLMAP world frame | {status} | {data.meters_per_colmap_unit:g} m / unit"


def plot_bounds(data, arrow_length):
    points = np.concatenate(list(data.centers.values()))
    low, high = points.min(axis=0), points.max(axis=0)
    padding = max(float(np.max(high - low)) * 0.06, arrow_length * 1.2, 0.01)
    return low - padding, high + padding


def frame_traces(data: TrajectoryData, index: int, arrow_length: float):
    traces = []
    for camera in CAMERAS:
        center = data.centers[camera][index]
        direction = data.forward[camera][index]
        tip = center + direction * arrow_length
        traces.append(go.Scatter3d(
            x=[center[0], tip[0]], y=[center[1], tip[1]], z=[center[2], tip[2]],
            mode="lines+markers", line=dict(color=COLORS[camera], width=7),
            marker=dict(size=[6, 0], color=COLORS[camera], symbol=SYMBOLS[camera]),
            name=LABELS[camera], legendgroup=camera, showlegend=False,
            text=[data.image_names[index]] * 2,
            hovertemplate=LABELS[camera] + "<br>%{text}<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>Z=%{z:.4f}<extra>当前帧 / 光轴 +Z</extra>",
        ))
        traces.append(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[direction[0]], v=[direction[1]], w=[direction[2]],
            sizemode="absolute", sizeref=arrow_length * 0.22, anchor="tip",
            colorscale=[[0, COLORS[camera]], [1, COLORS[camera]]],
            showscale=False, showlegend=False, legendgroup=camera, hoverinfo="skip",
        ))
    rig = [data.centers[camera][index] for camera in ("left", "mid", "right", "mid", "front")]
    rig = np.array(rig)
    traces.append(go.Scatter3d(x=rig[:, 0].tolist(), y=rig[:, 1].tolist(), z=rig[:, 2].tolist(),
                              mode="lines", line=dict(color="#64748b", width=3, dash="dot"),
                              showlegend=False, hoverinfo="skip"))
    return traces


def interactive_figure(data: TrajectoryData, arrow_length: float) -> go.Figure:
    figure = go.Figure()
    for camera in CAMERAS:
        points = data.centers[camera]
        figure.add_trace(go.Scatter3d(
            x=points[:, 0].tolist(), y=points[:, 1].tolist(), z=points[:, 2].tolist(),
            mode="lines+markers", name=LABELS[camera], legendgroup=camera,
            line=dict(color=COLORS[camera], width=3),
            marker=dict(color=COLORS[camera], size=3, symbol=SYMBOLS[camera]),
            customdata=[[i + 1, name] for i, name in enumerate(data.image_names)],
            hovertemplate=LABELS[camera] + "<br>帧 %{customdata[0]} · %{customdata[1]}"
                          "<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>",
        ))
        figure.add_trace(go.Scatter3d(
            x=points[[0, -1], 0].tolist(), y=points[[0, -1], 1].tolist(), z=points[[0, -1], 2].tolist(),
            mode="markers", name=LABELS[camera], legendgroup=camera, showlegend=False,
            marker=dict(color=COLORS[camera], size=7, symbol=["circle-open", "diamond-open"]),
            text=["起点", "终点"], hovertemplate="%{text}<extra>" + LABELS[camera] + "</extra>",
        ))
    indices = list(range(len(figure.data), len(figure.data) + 9))
    figure.add_traces(frame_traces(data, 0, arrow_length))
    figure.frames = [go.Frame(name=str(i + 1), traces=indices, data=frame_traces(data, i, arrow_length))
                     for i in range(data.frame_count)]
    low, high = plot_bounds(data, arrow_length)
    figure.update_layout(
        template="plotly_white", height=880, autosize=True,
        title=dict(text="四相机三维轨迹<br><sup>" + scale_caption(data, True) + "</sup>", x=0.03),
        font=dict(family="Microsoft YaHei, Arial, sans-serif", size=14),
        scene=dict(aspectmode="data", dragmode="orbit", uirevision="keep-view",
                   camera=dict(eye=dict(x=1.6, y=1.5, z=0.9)),
                   **{f"{axis}axis": dict(title=f"{axis.upper()} (COLMAP unit)",
                                           range=[float(low[i]), float(high[i])])
                      for i, axis in enumerate("xyz")}),
        legend=dict(orientation="h", x=0.02, y=1.01, groupclick="togglegroup"),
        margin=dict(l=10, r=10, t=120, b=145),
        sliders=[dict(active=0, x=0.06, len=0.9, y=-0.04,
                      currentvalue=dict(prefix="当前帧：", suffix=f" / {data.frame_count}"),
                      pad=dict(t=30), steps=[dict(label=str(i + 1), method="animate",
                          args=[[str(i + 1)], dict(mode="immediate", transition=dict(duration=0),
                                                  frame=dict(duration=0, redraw=True))])
                          for i in range(data.frame_count)])],
        annotations=[dict(x=0.02, y=-0.20, xref="paper", yref="paper", showarrow=False,
                          text="空心圆：起点　空心菱形：终点　箭头：当前帧相机光轴 +Z　|　拖动旋转 · 滚轮缩放 · 点击图例切换相机")],
    )
    return figure


def save_static_figure(data: TrajectoryData, output: Path, arrow_length: float, dpi: int):
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, width_ratios=[1, 1, 1.1])
    axis3d = figure.add_subplot(grid[:, :2], projection="3d")
    projections = [(figure.add_subplot(grid[i, 2]), pair) for i, pair in enumerate(((0, 1), (0, 2), (1, 2)))]
    low, high = plot_bounds(data, arrow_length)
    for camera in CAMERAS:
        points = data.centers[camera]
        color = COLORS[camera]
        axis3d.plot(*points.T, color=color, linewidth=1.7, label=camera)
        axis3d.scatter(*points[0], color=color, marker="o", s=42, edgecolors="white", zorder=4)
        axis3d.scatter(*points[-1], color=color, marker="X", s=55, edgecolors="white", zorder=4)
        # Show the optical directions at the starting poses; dynamic poses are in HTML.
        axis3d.quiver(*points[0], *data.forward[camera][0], length=arrow_length,
                      color=color, linewidth=1.5, arrow_length_ratio=0.25)
        for axis, (a, b) in projections:
            axis.plot(points[:, a], points[:, b], color=color, linewidth=1.3)
            axis.scatter(points[0, a], points[0, b], color=color, marker="o", s=24, edgecolors="white")
            axis.scatter(points[-1, a], points[-1, b], color=color, marker="X", s=32, edgecolors="white")
    for axis, (a, b) in projections:
        axis.set(xlabel=f"{'XYZ'[a]} (COLMAP unit)", ylabel=f"{'XYZ'[b]} (COLMAP unit)",
                 title=f"{'XYZ'[a]}{'XYZ'[b]} projection", xlim=(low[a], high[a]), ylim=(low[b], high[b]))
        # Expand each projection to its panel so small offsets remain readable.
        # The 3D view retains equal physical units; 2D scales are labeled explicitly.
        axis.set_aspect("auto")
        axis.grid(alpha=0.22)
        axis.tick_params(labelsize=9)
    axis3d.set(xlabel="X (COLMAP unit)", ylabel="Y (COLMAP unit)", zlabel="Z (COLMAP unit)",
               xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    axis3d.set_box_aspect(high - low)
    axis3d.view_init(elev=24, azim=-56)
    axis3d.legend(loc="upper left", frameon=False)
    axis3d.set_title("3D trajectories | circle: start | X: end | arrow: initial optical +Z", pad=22)
    figure.suptitle("Four-camera trajectories\n" + scale_caption(data)
                   + "\n3D: equal spatial units | 2D projections: independent axis scales", fontsize=14)
    figure.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(figure)


def visualize_poses(csv_path: Path, output_dir: Path, *, arrow_length=None, dpi=180):
    data = load_trajectories(csv_path)
    points = np.concatenate(list(data.centers.values()))
    if arrow_length is None:
        arrow_length = max(float(np.max(np.ptp(points, axis=0))) * 0.045, 0.02)
    if not math.isfinite(arrow_length) or arrow_length <= 0:
        raise ValueError("arrow_length must be finite and positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    html = output_dir / "camera_trajectories.html"
    png = output_dir / "camera_trajectories.png"
    summary_csv = output_dir / "trajectory_summary.csv"
    figure = interactive_figure(data, arrow_length)
    figure.write_html(str(html), include_plotlyjs=True, full_html=True, auto_play=False,
                      config=dict(responsive=True, scrollZoom=True, displaylogo=False))
    save_static_figure(data, png, arrow_length, dpi)
    summary = trajectory_summary(data)
    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(scale_caption(data))
    for row in summary:
        print(f"{row['camera']:5s}: {row['frame_count']} poses, sampled path length "
              f"{row['path_length']:.4f} COLMAP units")
    for path in (html, png, summary_csv):
        print(f"Saved: {path.resolve()}")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "outputs/poses.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--arrow-length", type=float, help="Optical-axis arrow length in COLMAP units")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    try:
        visualize_poses(args.csv, args.output, arrow_length=args.arrow_length, dpi=args.dpi)
    except (OSError, ValueError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
