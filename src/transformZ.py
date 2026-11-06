"""Restore metric scale and define the world frame by a reference front camera.

The callable API is align_model(); main() is only a standalone CLI adapter.
X_new = s * (R_ref @ X_old + t_ref), with output coordinates in meters.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.colmap_io import ImagePose, make_image_pose, read_front_poses, read_images_text
from src.distance_io import (
    convert_distance_excel_to_txt,
    iter_excel_distance_rows,
    read_distances,
)
from src.geometry import (
    camera_center,
    camera_forward_world,
    quaternion_to_rotation as qvec_to_rotmat,
    rotation_to_quaternion,
)


@dataclass(frozen=True)
class AlignmentConfig:
    input_model: Path
    output_model: Path
    distance_file: Path
    reference_image: str | None = None
    colmap_bin: str = "colmap"
    front_name_pattern: str = "front_*"
    front_camera_id: int | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class AlignmentResult:
    output_model: Path
    reference_image: str
    meters_per_input_unit: float
    world_transform: np.ndarray
    scale_report: dict


def read_images_txt(path):
    """Compatibility adapter for the previous dictionary-based API."""
    return {image.name: {
        "image_id": image.image_id, "camera_id": image.camera_id,
        "qvec": rotation_to_quaternion(image.camera_from_world[:3, :3]),
        "tvec": image.camera_from_world[:3, 3].copy(),
    } for image in read_images_text(Path(path))}


def find_reference(images: list[ImagePose], requested_name: str | None) -> ImagePose:
    if not images:
        raise ValueError("No registered front cameras")
    if requested_name is None:
        return images[0]  # Natural capture-name order from read_front_poses.
    normalized = requested_name.replace("\\", "/")
    exact = [image for image in images if image.name.replace("\\", "/") == normalized]
    if exact:
        return exact[0]
    basename = PurePosixPath(normalized).name
    matches = [image for image in images
               if PurePosixPath(image.name.replace("\\", "/")).name == basename]
    if len(matches) != 1:
        raise ValueError(f"Reference front image is missing or ambiguous: {requested_name}")
    return matches[0]


def estimate_metric_scale(images, distances: dict[str, float]) -> tuple[float, dict]:
    """Fit cumulative camera-path chords to cumulative meter readings.

    Every selected front image needs a distance. Unregistered distance rows are
    ignored. This preserves the previous transformZ least-squares scale fit.
    """
    if isinstance(images, dict):
        images = [make_image_pose(pose["image_id"], pose["camera_id"], name,
                                  pose["qvec"], pose["tvec"])
                  for name, pose in images.items()]
    by_basename = {}
    for name, distance in distances.items():
        by_basename.setdefault(name.rsplit("/", 1)[-1], []).append((name, distance))
    matched, used = [], set()
    for image in images:
        normalized = image.name.replace("\\", "/")
        if normalized in distances:
            distance_name, distance = normalized, distances[normalized]
        else:
            candidates = by_basename.get(normalized.rsplit("/", 1)[-1], [])
            if len(candidates) != 1:
                raise ValueError(f"Missing or ambiguous distance for image: {image.name}")
            distance_name, distance = candidates[0]
        if distance_name in used:
            raise ValueError(f"Distance record matches multiple cameras: {distance_name}")
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"Invalid distance for image: {image.name}")
        used.add(distance_name)
        matched.append((distance, image.name, image.center))
    if len(matched) < 2:
        raise ValueError("At least two registered front cameras with distances are required")
    matched.sort(key=lambda item: (item[0], item[1]))
    centers = np.array([item[2] for item in matched])
    readings = np.array([item[0] for item in matched])
    model_path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))]
    meter_path = readings - readings[0]
    if model_path[-1] <= 1e-12 or meter_path[-1] <= 0:
        raise ValueError("Camera trajectory and distance range must both be nonzero")
    scale = float(np.dot(model_path, meter_path) / np.dot(model_path, model_path))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Estimated scale must be finite and > 0")
    residuals = scale * model_path - meter_path
    return scale, {
        "scale_meters_per_model_unit": scale,
        "matched_images": len(matched),
        "unused_distance_records": len(distances) - len(used),
        "first_image": matched[0][1], "last_image": matched[-1][1],
        "first_distance_m": float(readings[0]), "last_distance_m": float(readings[-1]),
        "model_path_length": float(model_path[-1]),
        "distance_span_m": float(meter_path[-1]),
        "fit_rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
        "fit_max_abs_error_m": float(np.max(np.abs(residuals))),
        "method": "least_squares_cumulative_camera_path_chords",
    }


def build_world_transform(reference: ImagePose, scale: float) -> np.ndarray:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    transform = np.eye(4)
    transform[:3, :] = scale * reference.camera_from_world[:3, :]
    return transform


def write_transform_files(output: Path, reference: ImagePose, scale: float) -> Path:
    """COLMAP reads eight Sim3 values, not the homogeneous 4x4 matrix."""
    quaternion = rotation_to_quaternion(reference.camera_from_world[:3, :3])
    translation = scale * reference.camera_from_world[:3, 3]
    sim3_path = output / "transform_sim3.txt"
    np.savetxt(sim3_path, np.r_[scale, quaternion, translation][None, :], fmt="%.17g")
    np.savetxt(output / "T_new_from_old.txt", build_world_transform(reference, scale), fmt="%.17g")
    return sim3_path


def run_command(command):
    command = [str(value) for value in command]
    print("$ " + " ".join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    if result.stdout:
        print(result.stdout)
    if result.returncode:
        raise RuntimeError(f"COLMAP command failed ({result.returncode}): {' '.join(command)}")


def validate_output_path(output: Path, input_model: Path, distance_file: Path):
    """Protect source files before replacing an explicitly requested output."""
    if output == input_model or output in input_model.parents or input_model in output.parents:
        raise ValueError("Input and output model directories must be separate and non-nested")
    if output == distance_file or output in distance_file.parents:
        raise ValueError("Distance file must be outside the output model directory")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output model path is not a directory: {output}")


def verify_alignment(original, transformed, reference, scale, distances):
    after = {image.name: image for image in transformed}
    if set(after) != {image.name for image in original}:
        raise RuntimeError("Transformed model has different registered front images")
    reference_rotation = reference.camera_from_world[:3, :3]
    max_center_error = 0.0
    for image in original:
        actual = after[image.name]
        if (actual.image_id, actual.camera_id) != (image.image_id, image.camera_id):
            raise RuntimeError(f"Image identifiers changed: {image.name}")
        expected_center = scale * (reference_rotation @ image.center
                                   + reference.camera_from_world[:3, 3])
        expected_rotation = image.camera_from_world[:3, :3] @ reference_rotation.T
        if (not np.allclose(actual.center, expected_center, atol=1e-6, rtol=1e-6)
                or not np.allclose(actual.camera_from_world[:3, :3], expected_rotation,
                                   atol=1e-6, rtol=1e-6)):
            raise RuntimeError(f"World transform verification failed: {image.name}")
        max_center_error = max(max_center_error, float(np.linalg.norm(actual.center - expected_center)))
    if not np.allclose(after[reference.name].camera_from_world, np.eye(4), atol=1e-6, rtol=0):
        raise RuntimeError("Reference front camera is not the identity pose")
    verified_scale, _ = estimate_metric_scale(transformed, distances)
    if not np.isclose(verified_scale, 1.0, rtol=1e-6, atol=1e-8):
        raise RuntimeError("Transformed model failed metric scale verification")
    return {"reference_pose_identity": True, "output_meters_per_unit": verified_scale,
            "max_camera_center_error_m": max_center_error}


def align_model(config: AlignmentConfig) -> AlignmentResult:
    """Transform a complete sparse model and publish only after verification."""
    input_model = Path(config.input_model).resolve()
    output_model = Path(config.output_model).resolve()
    distance_file = Path(config.distance_file).resolve()
    validate_output_path(output_model, input_model, distance_file)
    if not input_model.is_dir():
        raise FileNotFoundError(input_model)
    if not distance_file.is_file():
        raise FileNotFoundError(distance_file)
    if output_model.exists() and any(output_model.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output model is not empty: {output_model}; use --overwrite")
    colmap_path = shutil.which(str(config.colmap_bin))
    if colmap_path is None:
        raise RuntimeError("COLMAP is required for model transformation. Install it or specify "
                           f"--colmap-bin with its executable path: {config.colmap_bin}")
    images, _ = read_front_poses(input_model, config.front_name_pattern, config.front_camera_id)
    reference = find_reference(images, config.reference_image)
    distances = read_distances(distance_file)
    scale, scale_report = estimate_metric_scale(images, distances)
    transform = build_world_transform(reference, scale)
    print(f"Reference: {reference.name}; metric scale: {scale:.12g} m/input unit")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colmap_align_", dir=output_model.parent) as temporary:
        staged = Path(temporary) / "model"
        staged.mkdir()
        transform_path = write_transform_files(staged, reference, scale)
        run_command([colmap_path, "model_transformer", "--input_path", input_model,
                     "--output_path", staged, "--transform_path", transform_path, "--is_inverse", "0"])
        transformed, _ = read_front_poses(staged, config.front_name_pattern, config.front_camera_id)
        scale_report.update({
            "input_model": str(input_model), "output_model": str(output_model),
            "distance_file": str(distance_file), "reference_image": reference.name,
            "length_unit": "m", "world_definition": "X_new = s * (R_ref @ X_old + t_ref)",
            "verification": verify_alignment(images, transformed, reference, scale, distances),
        })
        (staged / "scale_report.json").write_text(
            json.dumps(scale_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        validate_output_path(output_model, input_model, distance_file)
        if output_model.exists():
            if any(output_model.iterdir()) and not config.overwrite:
                raise FileExistsError(f"Output model became nonempty: {output_model}")
            shutil.rmtree(output_model)
        staged.replace(output_model)
    return AlignmentResult(output_model, reference.name, scale, transform, scale_report)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input COLMAP sparse model")
    parser.add_argument("--output", required=True, type=Path, help="Output metric sparse model")
    parser.add_argument("--distance", required=True, type=Path, help="Cumulative meter readings: TXT or Excel")
    parser.add_argument("--reference-image", help="Reference front image; default: first registered front image")
    parser.add_argument("--colmap-bin", default="colmap", help="COLMAP executable path or command")
    parser.add_argument("--front-name-pattern", default="front_*")
    parser.add_argument("--front-camera-id", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = align_model(AlignmentConfig(
            input_model=args.input, output_model=args.output, distance_file=args.distance,
            reference_image=args.reference_image, colmap_bin=args.colmap_bin,
            front_name_pattern=args.front_name_pattern, front_camera_id=args.front_camera_id,
            overwrite=args.overwrite,
        ))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Saved metric model: {result.output_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
