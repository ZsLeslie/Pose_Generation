"""Generate left/mid/right poses from front images and calibrated rig extrinsics.

Column vectors: X_A = H_A_from_B @ X_B. Quaternions use (w, x, y, z).
The input world frame is preserved. An already metric model uses 1 meter/unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import PROJECT_ROOT
from src.colmap_io import (
    ImagePose, make_image_pose, read_exact, read_front_poses, read_images_binary, read_images_text,
)
from src.geometry import invert_pose, quaternion_to_rotation, rotation_to_quaternion, validate_transform


CAMERAS = ("left", "mid", "right")
CSV_CAMERAS = ("front", *CAMERAS)


def load_extrinsics(path: Path, front_frame: str = "work"):
    if front_frame not in ("work", "capture"):
        raise ValueError("front_frame must be work or capture")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not storage.isOpened():
            raise ValueError(f"Cannot open OpenCV YAML: {path}")

        def read_matrix(key):
            node = storage.getNode(key)
            if node.empty():
                raise ValueError(f"Missing extrinsics matrix: {key}")
            return validate_transform(node.mat(), key)

        unit = storage.getNode("translation_unit").string()
        if unit != "m":
            raise ValueError(f"Expected translation_unit: m, got {unit!r}")
        reference = storage.getNode("reference_camera").string()
        if reference != "front":
            raise ValueError(f"Expected reference_camera: front, got {reference!r}")
        transforms = {camera: read_matrix(f"H_{camera}_from_front") for camera in CAMERAS}
        # These matrices ALREADY include the YAML's front-work pitch conversion.
        if front_frame == "capture":
            work_from_capture = read_matrix("H_front_work_from_front_capture")
            transforms = {name: matrix @ work_from_capture for name, matrix in transforms.items()}
        approximate = storage.getNode("extrinsics_approximate")
        metadata = {
            "translation_unit": unit, "reference_camera": reference,
            "front_pose": storage.getNode("front_pose").string(),
            "extrinsics_approximate": None if approximate.empty() else bool(approximate.real()),
            "accuracy_note": storage.getNode("accuracy_note").string(),
        }
        return transforms, metadata
    finally:
        storage.release()


def propagate_pose(front_from_world: np.ndarray, camera_from_front_m: np.ndarray,
                   meters_per_colmap_unit: float) -> np.ndarray:
    if not math.isfinite(meters_per_colmap_unit) or meters_per_colmap_unit <= 0:
        raise ValueError("meters_per_colmap_unit must be finite and positive")
    camera_from_front = np.asarray(camera_from_front_m, dtype=float).copy()
    camera_from_front[:3, 3] /= meters_per_colmap_unit
    return camera_from_front @ front_from_world


def pose_record(camera_from_world: np.ndarray) -> dict:
    world_from_camera = invert_pose(camera_from_world)
    return {
        "qvec_wxyz": rotation_to_quaternion(camera_from_world[:3, :3]).tolist(),
        "tvec": camera_from_world[:3, 3].tolist(),
        "camera_center_world": world_from_camera[:3, 3].tolist(),
        "camera_from_world": camera_from_world.tolist(),
        "world_from_camera": world_from_camera.tolist(),
    }


def compute_rig_poses(images: list[ImagePose], transforms: dict[str, np.ndarray], scale: float) -> list[dict]:
    frames = []
    for image in images:
        poses = {"front": pose_record(image.camera_from_world)}
        for camera in CAMERAS:
            poses[camera] = pose_record(propagate_pose(image.camera_from_world, transforms[camera], scale))
        frames.append({
            "source_image_id": image.image_id, "source_camera_id": image.camera_id,
            "front_image_name": image.name, "poses": poses,
        })
    return frames


def save_pose_files(result: dict, output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "poses.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    metadata = result["metadata"]
    with (output / "poses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "source_image_id", "source_camera_id", "front_image_name", "camera",
            "qw", "qx", "qy", "qz", "tx", "ty", "tz", "cx", "cy", "cz",
            "meters_per_colmap_unit", "scale_status", "front_frame", "length_unit",
        ])
        for frame in result["frames"]:
            for camera in CSV_CAMERAS:
                pose = frame["poses"][camera]
                writer.writerow([
                    frame["source_image_id"], frame["source_camera_id"], frame["front_image_name"], camera,
                    *pose["qvec_wxyz"], *pose["tvec"], *pose["camera_center_world"],
                    metadata["meters_per_colmap_unit"], metadata["scale_status"],
                    metadata["front_frame"], metadata["length_unit"],
                ])


def generate_poses(model: Path, extrinsics: Path, output: Path, *,
                   meters_per_colmap_unit: float | None = None,
                   front_frame: str = "work", front_name_pattern: str = "front_*",
                   front_camera_id: int | None = None, length_unit: str = "colmap_unit",
                   world_frame: str = "unchanged COLMAP world frame", alignment: dict | None = None) -> dict:
    scale = 1.0 if meters_per_colmap_unit is None else meters_per_colmap_unit
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("meters_per_colmap_unit must be finite and positive")
    if length_unit not in ("m", "colmap_unit"):
        raise ValueError("length_unit must be m or colmap_unit")
    if length_unit == "m" and (meters_per_colmap_unit is None or scale != 1.0):
        raise ValueError("A metric input model must explicitly use 1 meter per unit")
    model, extrinsics = Path(model), Path(extrinsics)
    transforms, calibration = load_extrinsics(extrinsics, front_frame)
    images, source = read_front_poses(model, front_name_pattern, front_camera_id)
    warnings = []
    if meters_per_colmap_unit is None:
        warnings.append("COLMAP scale is unknown: provisionally assuming 1 unit = 1 m; positions are provisional.")
    if calibration["extrinsics_approximate"]:
        warnings.append("The YAML declares approximate extrinsics: " + calibration["accuracy_note"])
    for warning in warnings:
        print("WARNING: " + warning, file=sys.stderr)
    metadata = {
        "source_images": str(source.resolve()), "extrinsics_file": str(extrinsics.resolve()),
        "front_frame": front_frame, "front_name_pattern": front_name_pattern,
        "front_camera_id_filter": front_camera_id, "frame_count": len(images),
        "generated_pose_count": len(images) * len(CAMERAS), "exported_pose_count": len(images) * len(CSV_CAMERAS),
        "world_frame": world_frame, "length_unit": length_unit, "meters_per_colmap_unit": scale,
        "scale_status": "assumed" if meters_per_colmap_unit is None else "provided",
        "quaternion_order": "wxyz", "quaternion_direction": "world_to_camera",
        "matrix_definition": "X_A = H_A_from_B @ X_B; column vectors",
        "calibration": calibration, "warnings": warnings,
    }
    if alignment is not None:
        metadata["alignment"] = alignment
    result = {"metadata": metadata, "frames": compute_rig_poses(images, transforms, scale)}
    save_pose_files(result, output)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "colmap/sparse")
    parser.add_argument("--extrinsics", type=Path, default=PROJECT_ROOT / "extrinsics/rig_extrinsics.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--meters-per-colmap-unit", type=float,
                        help="Meters per input unit; omission assumes 1 provisionally")
    parser.add_argument("--metric-model", action="store_true", help="Input model is already metric")
    parser.add_argument("--front-frame", choices=("work", "capture"), default="work")
    parser.add_argument("--front-name-pattern", default="front_*")
    parser.add_argument("--front-camera-id", type=int)
    args = parser.parse_args(argv)
    if args.metric_model and args.meters_per_colmap_unit not in (None, 1.0):
        parser.error("--metric-model requires 1 meter per model unit")
    try:
        result = generate_poses(
            args.model, args.extrinsics, args.output,
            meters_per_colmap_unit=1.0 if args.metric_model else args.meters_per_colmap_unit,
            front_frame=args.front_frame, front_name_pattern=args.front_name_pattern,
            front_camera_id=args.front_camera_id, length_unit="m" if args.metric_model else "colmap_unit",
        )
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Read {result['metadata']['frame_count']} front poses; exported "
          f"{result['metadata']['exported_pose_count']} four-camera poses to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
