"""Restore metric scale and define the world frame by a reference front camera.

The callable API is align_model(); main() is only a standalone CLI adapter.
X_new = s * (R_ref @ X_old + t_ref), with output coordinates in meters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.colmap_io import ImagePose, read_front_poses, read_images_text
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
from src.scale_estimation import ScaleFitOptions, estimate_metric_scale


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
    scale_fit_options: ScaleFitOptions = ScaleFitOptions()


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


def verify_alignment(original, transformed, reference, scale, distances, *,
                     fit_image_names=None, scale_fit_options=None):
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
    options = ScaleFitOptions() if scale_fit_options is None else scale_fit_options
    if fit_image_names is None:
        _, original_fit = estimate_metric_scale(original, distances, options=options)
        fit_image_names = original_fit["fit_image_names"]
    selected_names = set(fit_image_names)
    if len(selected_names) < 2 or not selected_names.issubset(after):
        raise RuntimeError("Invalid image set for metric scale verification")
    # Keep the exact original fit anchors. Re-detecting/filtering after the
    # transform could conceal a changed fit set or a previously contaminated fit.
    fit_images = [image for image in transformed if image.name in selected_names]
    verified_scale, output_fit = estimate_metric_scale(
        fit_images, distances, options=replace(options, filter_position_outliers=False))
    if not np.isclose(verified_scale, 1.0, rtol=1e-6, atol=1e-8):
        raise RuntimeError("Transformed model failed metric scale verification")
    return {"reference_pose_identity": True, "output_meters_per_unit": verified_scale,
            "max_camera_center_error_m": max_center_error,
            "scale_fit_images": len(fit_images), "scale_fit_quality_passed": True,
            "distance_fit_rmse_m": output_fit["fit_rmse_m"]}


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
    images, images_source = read_front_poses(input_model, config.front_name_pattern, config.front_camera_id)
    reference = find_reference(images, config.reference_image)
    distances = read_distances(distance_file)
    scale, scale_report = estimate_metric_scale(images, distances, options=config.scale_fit_options)
    if reference.name not in scale_report["fit_image_names"]:
        raise ValueError(f"Reference image was excluded from the scale fit: {reference.name}; "
                         "choose a reliable frame with --reference-image")
    scale_report["input_images_sha256"] = hashlib.sha256(images_source.read_bytes()).hexdigest()
    for record in scale_report["excluded_pose_outliers"]:
        print(f"WARNING: Excluded from scale fitting: {record['image']} "
              f"(position residual {record['position_error_m']:.6g} m)", file=sys.stderr)
    transform = build_world_transform(reference, scale)
    print(f"Reference: {reference.name}; metric scale: {scale:.12g} m/input unit")
    print(f"Scale fit: {scale_report['fit_images']}/{scale_report['matched_images']} frames; "
          f"RMSE {scale_report['fit_rmse_m']:.6g} m "
          f"({scale_report['fit_relative_rmse']:.2%} of distance span)")

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
            "verification": verify_alignment(
                images, transformed, reference, scale, distances,
                fit_image_names=scale_report["fit_image_names"], scale_fit_options=config.scale_fit_options),
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
