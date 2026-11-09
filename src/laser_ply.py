"""Stitch LaserScan JSONL records using an already metric sparse reconstruction.

Adapted from the supplied laser_ply.py: distance-based interpolation, SLERP,
robust pose filtering, intensity colors and sensor-local polar coordinates.
The input model's world frame and metric scale are used without modification.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import PROJECT_ROOT
from src.colmap_io import read_front_poses
from src.distance_io import EXCEL_DISTANCE_SUFFIXES, IMAGE_NAME_HEADERS, iter_excel_distance_rows, read_distances
from src.geometry import invert_pose, validate_transform
from src.pointcloud import radius_outlier_filter, statistical_outlier_filter, voxel_downsample, write_ply


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaserConfig:
    metric_model: Path
    distance_file: Path
    scan_file: Path
    output_dir: Path
    extrinsics_file: Path | None = None
    front_name_pattern: str = "front_*"
    front_camera_id: int | None = None
    front_frame: str = "work"
    distance_step_m: float = 0.01
    time_margin_s: float = 20.0
    min_range_m: float = 0.05
    max_range_m: float = 50.0
    filter_pose_outliers: bool = True
    pose_min_position_m: float = 0.15
    pose_min_rotation_deg: float = 2.0
    pose_mad_factor: float = 6.0
    pose_max_iterations: int = 5
    # The supplied script has these point-cloud filters commented out.
    voxel_size_m: float = 0.0
    sor_neighbors: int = 0
    sor_std_ratio: float = 3.0
    ror_radius_m: float = 0.0
    ror_min_neighbors: int = 2
    overwrite: bool = False


@dataclass(frozen=True)
class LaserResult:
    ply_path: Path
    poses_path: Path
    report_path: Path
    scan_count: int
    used_scan_count: int
    point_count: int
    excluded_pose_names: tuple[str, ...]


@dataclass(frozen=True)
class PoseAnchor:
    image_id: int
    name: str
    distance: float
    timestamp: float
    rotation: np.ndarray  # World -> front camera.
    center: np.ndarray   # Metric sparse world coordinates.


@dataclass(frozen=True)
class ScanFrame:
    line_number: int
    timestamp: float
    distance: float
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray
    intensities: np.ndarray | None


def timestamp_seconds(value):
    """Accept epoch seconds, milliseconds, microseconds or nanoseconds."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Timestamp must be finite and nonnegative")
    for threshold, divisor in ((1e17, 1e9), (1e14, 1e6), (1e11, 1e3)):
        if value >= threshold:
            return value / divisor
    return value


def timestamp_from_name(name):
    stem = PurePosixPath(name.replace("\\", "/")).stem
    tokens = re.findall(r"\d+(?:\.\d+)?", stem)
    if not tokens or len(tokens[-1].split(".")[0]) < 10:
        raise ValueError(f"No capture timestamp in {name}; supply an Excel time/timestamp column")
    return timestamp_seconds(tokens[-1])


def read_image_times(distance_file):
    """Excel time is optional; TXT inputs use the capture timestamp in the name."""
    if Path(distance_file).suffix.lower() not in EXCEL_DISTANCE_SUFFIXES:
        return {}
    rows = iter_excel_distance_rows(Path(distance_file))
    try:
        for _, values in rows:
            if not any(value is not None and str(value).strip() for value in values):
                continue
            headers = [str(value).strip().lower() for value in values]
            names = [i for i, value in enumerate(headers) if value in IMAGE_NAME_HEADERS]
            times = [i for i, value in enumerate(headers) if value in {"time", "timestamp", "timestamp_s", "时间", "时间戳"}]
            if len(names) != 1 or len(times) != 1:
                return {}
            name_column, time_column = names[0], times[0]
            break
        else:
            return {}
        result = {}
        for line_number, values in rows:
            if len(values) <= max(name_column, time_column):
                continue
            name, value = values[name_column], values[time_column]
            if name is None or value is None or str(value).strip() == "":
                continue
            try:
                result[str(name).strip().replace("\\", "/")] = timestamp_seconds(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{distance_file}:row {line_number}: invalid capture timestamp") from error
        return result
    finally:
        rows.close()


def match_record(name, values):
    normalized = name.replace("\\", "/")
    if normalized in values:
        return normalized, values[normalized]
    matches = [(key, value) for key, value in values.items()
               if key.rsplit("/", 1)[-1] == normalized.rsplit("/", 1)[-1]]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous distance/time record for {name}")
    return matches[0]


def load_pose_anchors(config):
    images, _ = read_front_poses(config.metric_model, config.front_name_pattern, config.front_camera_id)
    distances = read_distances(config.distance_file)
    capture_times = read_image_times(config.distance_file)
    records, used_names = [], set()
    for image in images:
        key, distance = match_record(image.name, distances)
        if key in used_names:
            raise ValueError(f"Distance record matches multiple cameras: {key}")
        used_names.add(key)
        if capture_times:
            _, timestamp = match_record(image.name, capture_times)
        else:
            timestamp = timestamp_from_name(image.name)
        records.append(PoseAnchor(image.image_id, image.name, distance, timestamp,
                                  image.camera_from_world[:3, :3].copy(), image.center))
    records.sort(key=lambda record: (record.timestamp, record.image_id))
    anchors, skipped = [], []
    for record in records:
        if anchors and record.timestamp <= anchors[-1].timestamp:
            raise ValueError("Front image capture timestamps must be unique and increasing")
        if anchors and record.distance < anchors[-1].distance - 1e-9:
            skipped.append({"image": record.name, "reason": "decreasing distance"})
            continue
        if anchors and abs(record.distance - anchors[-1].distance) <= 1e-9:
            skipped.append({"image": anchors[-1].name, "reason": "duplicate distance; keeping last pose"})
            anchors[-1] = record
        else:
            anchors.append(record)
    return anchors, skipped


def filter_pose_outliers(anchors, config):
    """Preserve the source script's iterative MAD thresholds and local peaks."""
    records, removed = list(anchors), []
    if not config.filter_pose_outliers:
        return records, removed
    for iteration in range(config.pose_max_iterations):
        if len(records) < 5:
            break
        errors = []
        for index in range(1, len(records) - 1):
            previous, current, following = records[index - 1:index + 2]
            span = following.distance - previous.distance
            if span <= 1e-9:
                continue
            alpha = float(np.clip((current.distance - previous.distance) / span, 0, 1))
            predicted_center = (1 - alpha) * previous.center + alpha * following.center
            interpolator = Slerp([0., 1.], Rotation.from_matrix([previous.rotation, following.rotation]))
            angular_error = (interpolator(alpha).inv() * Rotation.from_matrix(current.rotation)).magnitude()
            errors.append((index, float(np.linalg.norm(current.center - predicted_center)),
                           float(np.degrees(angular_error))))
        if not errors:
            break

        def threshold(values, minimum):
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            return max(minimum, float(median + config.pose_mad_factor * 1.4826 * mad))

        position_limit = threshold(np.array([row[1] for row in errors]), config.pose_min_position_m)
        rotation_limit = threshold(np.array([row[2] for row in errors]), config.pose_min_rotation_deg)
        scores = {i: max(position / position_limit, angle / rotation_limit) for i, position, angle in errors}
        candidates = {i for i, score in scores.items() if score > 1
                      and score >= scores.get(i - 1, -np.inf) and score >= scores.get(i + 1, -np.inf)}
        if not candidates:
            break
        for index, position, angle in errors:
            if index in candidates:
                record = records[index]
                removed.append({"image_id": record.image_id, "image": record.name,
                                "distance_m": record.distance, "iteration": iteration + 1,
                                "position_error_m": position, "rotation_error_deg": angle,
                                "position_threshold_m": position_limit, "rotation_threshold_deg": rotation_limit})
                LOGGER.warning("Excluded from laser interpolation: %s (position %.3fm, rotation %.3fdeg)",
                               record.name, position, angle)
        records = [record for index, record in enumerate(records) if index not in candidates]
    return records, removed


def load_scans(path):
    """Read scan.json as JSON Lines, converting one record at a time to arrays."""
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                ranges = np.asarray(data["ranges"], dtype=np.float64)
                intensities = (None if data.get("intensities") is None or len(data["intensities"]) == 0
                               else np.asarray(data["intensities"], dtype=np.float64))
                if ranges.ndim != 1 or not len(ranges):
                    raise ValueError("ranges must be a nonempty one-dimensional array")
                if intensities is not None and intensities.shape != ranges.shape:
                    raise ValueError("intensities and ranges must have equal lengths")
                distance = float(data["distance"])
                angle_min, increment = float(data["angle_min"]), float(data["angle_increment"])
                low, high = float(data.get("range_min", 0)), float(data.get("range_max", 50))
                if (not all(math.isfinite(value) for value in (distance, angle_min, increment, low, high))
                        or distance < 0 or increment == 0 or low < 0 or high <= low):
                    raise ValueError("Invalid scan distance, angles or range limits")
                records.append(ScanFrame(line_number, timestamp_seconds(data["timestamp"]), distance,
                                         angle_min, increment, low, high, ranges, intensities))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid LaserScan record: {error}") from error
    if not records:
        raise ValueError(f"No laser scans in {path}")
    return sorted(records, key=lambda record: (record.timestamp, record.line_number))


def load_laser_extrinsics(path=None):
    """H_front_from_laser maps source-script laser-local coordinates into front."""
    if path is None:
        matrix = np.array([[1., 0, 0, -0.036], [0, 0, -1, 0.032258],
                           [0, 1, 0, -0.4425], [0, 0, 0, 1]])
        return matrix, "supplied laser_ply.py: offset_mm=[-36,32.258,-442.5], Rx(90deg)"
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not storage.isOpened():
            raise ValueError(f"Cannot open laser extrinsics: {path}")
        if storage.getNode("translation_unit").string() != "m":
            raise ValueError("Laser extrinsics must declare translation_unit: m")
        node = storage.getNode("H_front_from_laser")
        if node.empty():
            raise ValueError("Laser extrinsics must contain H_front_from_laser")
        return validate_transform(node.mat(), "H_front_from_laser"), str(path.resolve())
    finally:
        storage.release()


def select_scan_indices(scans, timestamps, first, second, config, used):
    start = np.searchsorted(timestamps, first.timestamp - config.time_margin_s, side="left")
    stop = np.searchsorted(timestamps, second.timestamp + config.time_margin_s, side="right")
    # Reject backward readings against the last retained reading; keep the last
    # scan at a repeated distance, as in the original script.
    by_distance, last_distance = {}, -np.inf
    for index in range(int(start), int(stop)):
        record = scans[index]
        if record.distance < last_distance - 1e-9:
            continue
        last_distance = max(last_distance, record.distance)
        if first.distance - 1e-9 <= record.distance <= second.distance + 1e-9:
            by_distance[record.distance] = index
    distances = np.array(sorted(by_distance))
    if not len(distances):
        return []
    if config.distance_step_m == 0:
        candidates = [by_distance[float(distance)] for distance in distances]
    else:
        span = second.distance - first.distance
        count = int(math.floor(span / config.distance_step_m))
        targets = np.r_[first.distance + np.arange(count + 1) * config.distance_step_m, second.distance]
        positions = np.searchsorted(distances, targets)
        right = np.clip(positions, 0, len(distances) - 1)
        left = np.clip(positions - 1, 0, len(distances) - 1)
        nearest = np.where(np.abs(distances[right] - targets) < np.abs(distances[left] - targets), right, left)
        candidates = [by_distance[float(distances[position])] for position in np.unique(nearest)]
    return [index for index in candidates if index not in used]


def scan_points(scan, config):
    lower, upper = max(config.min_range_m, scan.range_min), min(config.max_range_m, scan.range_max)
    valid = np.isfinite(scan.ranges) & (scan.ranges >= lower) & (scan.ranges <= upper)
    ranges = scan.ranges[valid]
    angles = scan.angle_min + scan.angle_increment * np.flatnonzero(valid)
    # Source convention, not the usual ROS XY scan-plane convention.
    points = np.column_stack([ranges * np.sin(angles), np.zeros(len(ranges)), -ranges * np.cos(angles)])
    colors = np.full((len(ranges), 3), 200, dtype=np.uint8)
    if scan.intensities is not None:
        gray = np.nan_to_num(scan.intensities[valid], nan=200., posinf=255., neginf=0.)
        colors[:] = np.clip(gray, 0, 255).astype(np.uint8)[:, None]
    return points, colors


def generate_laser_points(anchors, scans, front_from_laser, config):
    timestamps = np.array([scan.timestamp for scan in scans])
    used, positions, colors, poses = set(), [], [], []
    for first, second in zip(anchors, anchors[1:]):
        span = second.distance - first.distance
        if span <= 1e-9:
            continue
        interpolator = Slerp([0., 1.], Rotation.from_matrix([first.rotation, second.rotation]))
        for index in select_scan_indices(scans, timestamps, first, second, config, used):
            used.add(index)
            scan = scans[index]
            # Use the actual chosen scan distance, not the requested grid point.
            alpha = float(np.clip((scan.distance - first.distance) / span, 0, 1))
            rotation = interpolator(alpha).as_matrix()
            center = (1 - alpha) * first.center + alpha * second.center
            world_from_front = np.eye(4)
            world_from_front[:3, :3], world_from_front[:3, 3] = rotation.T, center
            world_from_laser = world_from_front @ front_from_laser
            local, scan_colors = scan_points(scan, config)
            if not len(local):
                continue
            positions.append(local @ world_from_laser[:3, :3].T + world_from_laser[:3, 3])
            colors.append(scan_colors)
            poses.append({
                "scan_line": scan.line_number, "timestamp_s": scan.timestamp, "distance_m": scan.distance,
                "previous_front_image": first.name, "next_front_image": second.name, "alpha": alpha,
                "valid_point_count": len(local), "world_from_laser": world_from_laser.tolist(),
                "laser_from_world": invert_pose(world_from_laser).tolist(),
            })
    if not positions:
        raise ValueError("No laser points matched the front trajectory; check timestamps, distances and ranges")
    return np.concatenate(positions), np.concatenate(colors), poses, len(used)


def validate_config(config):
    if config.front_frame not in ("work", "capture"):
        raise ValueError("Laser front_frame must be work or capture")
    nonnegative = (config.distance_step_m, config.time_margin_s, config.voxel_size_m,
                   config.sor_std_ratio, config.ror_radius_m, config.min_range_m)
    positive = (config.max_range_m, config.pose_min_position_m,
                config.pose_min_rotation_deg, config.pose_mad_factor)
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("Laser steps, margins, ranges and filter sizes must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("Laser ranges and pose thresholds must be finite and positive")
    if config.max_range_m <= config.min_range_m:
        raise ValueError("Laser max range must exceed min range")
    if config.sor_neighbors < 0 or config.ror_min_neighbors < 1 or config.pose_max_iterations < 1:
        raise ValueError("Invalid laser neighbor or iteration count")
    output = Path(config.output_dir).resolve()
    sources = [Path(config.metric_model).resolve(), Path(config.distance_file).resolve(), Path(config.scan_file).resolve()]
    if config.extrinsics_file is not None:
        sources.append(Path(config.extrinsics_file).resolve())
    for source in sources:
        if output == source or output in source.parents:
            raise ValueError(f"Laser output directory must not contain a source: {source}")
    if sources[0] in output.parents:
        raise ValueError("Laser output must be outside the metric sparse model")
    if not sources[0].is_dir():
        raise FileNotFoundError(sources[0])
    for source in sources[1:]:
        if not source.is_file():
            raise FileNotFoundError(source)
    if output.exists() and not output.is_dir():
        raise ValueError(f"Laser output is not a directory: {output}")
    if not config.overwrite and any((output / name).exists() for name in
                                     ("laser.ply", "laser_poses.json", "laser_report.json")):
        raise FileExistsError(f"Laser results exist in {output}; use --overwrite")


def stitch_laser_scans(config: LaserConfig) -> LaserResult:
    """Consume a metric model directly; no pycolmap, axis flattening or rescaling."""
    validate_config(config)
    anchors, skipped = load_pose_anchors(config)
    original_count = len(anchors)
    anchors, excluded = filter_pose_outliers(anchors, config)
    if len(anchors) < 2:
        raise ValueError("Laser interpolation requires two distinct retained front-camera distances")
    scans = load_scans(config.scan_file)
    front_from_laser, extrinsics_source = load_laser_extrinsics(config.extrinsics_file)
    points, colors, poses, selected_count = generate_laser_points(anchors, scans, front_from_laser, config)
    raw_count = len(points)
    points, colors = voxel_downsample(points, colors, config.voxel_size_m)
    points, colors = statistical_outlier_filter(points, colors, config.sor_neighbors, config.sor_std_ratio)
    points, colors = radius_outlier_filter(points, colors, config.ror_radius_m, config.ror_min_neighbors)
    if not len(points):
        raise ValueError("Laser point-cloud filters removed every point")
    output = Path(config.output_dir).resolve()
    paths = [output / name for name in ("laser.ply", "laser_poses.json", "laser_report.json")]
    report = {
        "metric_model": str(Path(config.metric_model).resolve()), "scan_file": str(Path(config.scan_file).resolve()),
        "distance_file": str(Path(config.distance_file).resolve()), "length_unit": "m",
        "front_frame": config.front_frame,
        "world_frame": "unchanged metric sparse world frame", "additional_scale": 1.0,
        "laser_local_definition": "[range*sin(angle), 0, -range*cos(angle)]",
        "H_front_from_laser": front_from_laser.tolist(), "laser_extrinsics_source": extrinsics_source,
        "world_definition": "X_world = H_world_from_front @ H_front_from_laser @ X_laser",
        "input_pose_count": original_count + len(skipped), "retained_pose_count": len(anchors),
        "skipped_pose_records": skipped, "excluded_pose_outliers": excluded,
        "scan_count": len(scans), "selected_scan_count": selected_count,
        "used_scan_count": len(poses), "unselected_scan_count": len(scans) - selected_count,
        "raw_point_count": raw_count, "point_count": len(points),
        "distance_span_m": [anchors[0].distance, anchors[-1].distance],
        "bounds_world_m": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "ply_path": str(paths[0]), "poses_path": str(paths[1]),
        "notes": ["Pose outliers are excluded only from laser interpolation; sparse and rig poses are unchanged.",
                  "Default laser mounting parameters are inherited from the supplied script; verify them for this rig.",
                  "No per-beam motion compensation is applied within an individual scan."],
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="laser_build_", dir=output) as temporary:
        staged = Path(temporary)
        write_ply(points, colors, staged / paths[0].name)
        (staged / paths[1].name).write_text(json.dumps({"metadata": report, "frames": poses},
                                                      indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (staged / paths[2].name).write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                     allow_nan=False) + "\n", encoding="utf-8")
        if not config.overwrite and any(path.exists() for path in paths):
            raise FileExistsError(f"Laser output files appeared while building: {output}")
        for path in paths:
            (staged / path.name).replace(path)
    LOGGER.info("Stitched %d scans into %d metric laser points: %s", len(poses), len(points), paths[0])
    return LaserResult(*paths, len(scans), len(poses), len(points), tuple(row["image"] for row in excluded))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Already metric sparse model")
    parser.add_argument("--distance", type=Path, default=PROJECT_ROOT / "distance.xlsx")
    parser.add_argument("--scan", type=Path, default=PROJECT_ROOT / "scan/scan.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/laser")
    parser.add_argument("--laser-extrinsics", type=Path, help="Optional H_front_from_laser OpenCV YAML, in meters")
    parser.add_argument("--distance-step", type=float, default=0.01, help="Distance grid in meters; 0 selects all unique distances")
    parser.add_argument("--front-name-pattern", default="front_*")
    parser.add_argument("--front-camera-id", type=int)
    parser.add_argument("--front-frame", choices=("work", "capture"), default="work",
                        help="Front optical frame in which the laser mounting extrinsics are expressed")
    parser.add_argument("--keep-pose-outliers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = stitch_laser_scans(LaserConfig(
            metric_model=args.model, distance_file=args.distance, scan_file=args.scan, output_dir=args.output,
            extrinsics_file=args.laser_extrinsics, distance_step_m=args.distance_step,
            front_name_pattern=args.front_name_pattern, front_camera_id=args.front_camera_id,
            front_frame=args.front_frame,
            filter_pose_outliers=not args.keep_pose_outliers, overwrite=args.overwrite,
        ))
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Laser cloud: {result.ply_path} ({result.point_count} points, {result.used_scan_count} scans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
