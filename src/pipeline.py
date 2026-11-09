"""Metric reference-camera alignment, laser stitching and camera rig poses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import PROJECT_ROOT
from .generate_poses import generate_poses, load_extrinsics
from .laser_ply import LaserConfig, load_laser_extrinsics, stitch_laser_scans
from .transformZ import AlignmentConfig, align_model


@dataclass(frozen=True)
class PipelineConfig:
    input_model: Path = PROJECT_ROOT / "colmap/sparse"
    distance_file: Path = PROJECT_ROOT / "distance.xlsx"
    extrinsics_file: Path = PROJECT_ROOT / "extrinsics/rig_extrinsics.yaml"
    output_dir: Path = PROJECT_ROOT / "outputs/pipeline"
    reference_image: str | None = None
    colmap_bin: str = "colmap"
    front_frame: str = "work"
    front_name_pattern: str = "front_*"
    front_camera_id: int | None = None
    overwrite: bool = False
    scan_file: Path | None = PROJECT_ROOT / "scan/scan.json"
    laser_extrinsics_file: Path | None = None
    laser_distance_step_m: float = 0.01
    laser_filter_pose_outliers: bool = True


@dataclass(frozen=True)
class PipelineResult:
    aligned_model: Path
    poses_json: Path
    poses_csv: Path
    report_path: Path
    reference_image: str
    frame_count: int
    meters_per_original_unit: float
    laser_ply: Path | None = None
    laser_poses: Path | None = None
    laser_report: Path | None = None


def validate_pipeline_paths(input_model, distance_file, extrinsics_file, output_dir, overwrite,
                            scan_file=None, laser_extrinsics_file=None):
    if not input_model.is_dir():
        raise FileNotFoundError(input_model)
    source_files = [distance_file, extrinsics_file]
    if scan_file is not None:
        source_files.append(scan_file)
        if laser_extrinsics_file is not None:
            source_files.append(laser_extrinsics_file)
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Overwrite applies only to generated results; source data stays separate.
    for source in (input_model, *source_files):
        if output_dir == source or output_dir in source.parents:
            raise ValueError(f"Output directory must not contain a source: {source}")
    if input_model in output_dir.parents:
        raise ValueError("Output directory must not be inside the input sparse model")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    generated = [output_dir / "sparse_metric", output_dir / "poses", output_dir / "pipeline_report.json"]
    if scan_file is not None:
        generated.append(output_dir / "laser")
    if not overwrite and any(path.exists() for path in generated):
        raise FileExistsError(f"Pipeline results already exist in {output_dir}; use --overwrite")


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Keep CLI parsing out of the workflow so other Python code can reuse it."""
    input_model = Path(config.input_model).resolve()
    distance_file = Path(config.distance_file).resolve()
    extrinsics_file = Path(config.extrinsics_file).resolve()
    output_dir = Path(config.output_dir).resolve()
    scan_file = None if config.scan_file is None else Path(config.scan_file).resolve()
    laser_extrinsics = (None if config.laser_extrinsics_file is None
                        else Path(config.laser_extrinsics_file).resolve())
    validate_pipeline_paths(input_model, distance_file, extrinsics_file, output_dir, config.overwrite,
                            scan_file, laser_extrinsics)
    # Reject invalid calibration before starting an external model transformation.
    load_extrinsics(extrinsics_file, config.front_frame)
    if scan_file is not None:
        if not math.isfinite(config.laser_distance_step_m) or config.laser_distance_step_m < 0:
            raise ValueError("Laser distance step must be finite and nonnegative")
        load_laser_extrinsics(laser_extrinsics)

    step_count = 2 if scan_file is None else 3
    print(f"[1/{step_count}] Restore metric scale and align the reference-camera world frame")
    aligned = align_model(AlignmentConfig(
        input_model=input_model, output_model=output_dir / "sparse_metric",
        distance_file=distance_file, reference_image=config.reference_image,
        colmap_bin=config.colmap_bin, front_name_pattern=config.front_name_pattern,
        front_camera_id=config.front_camera_id, overwrite=config.overwrite,
    ))

    laser_result = None
    if scan_file is not None:
        print("[2/3] Stitch laser scans using the metric sparse front-camera trajectory")
        laser_result = stitch_laser_scans(LaserConfig(
            metric_model=aligned.output_model, distance_file=distance_file,
            scan_file=scan_file, output_dir=output_dir / "laser", extrinsics_file=laser_extrinsics,
            front_name_pattern=config.front_name_pattern, front_camera_id=config.front_camera_id,
            front_frame=config.front_frame,
            distance_step_m=config.laser_distance_step_m,
            filter_pose_outliers=config.laser_filter_pose_outliers, overwrite=config.overwrite,
        ))

    print(f"[{step_count}/{step_count}] Compute left/mid/right camera poses from the metric front-camera poses")
    alignment_info = {
        "original_model": str(input_model), "reference_image": aligned.reference_image,
        "original_meters_per_colmap_unit": aligned.meters_per_input_unit,
        "world_transform_new_from_old": aligned.world_transform.tolist(),
        "scale_report": str(aligned.output_model / "scale_report.json"),
    }
    poses_dir = output_dir / "poses"
    poses = generate_poses(
        aligned.output_model, extrinsics_file, poses_dir,
        # Stage 1 has ALREADY applied the original scale to the whole model.
        # Calibration translations and input poses are now both in meters.
        meters_per_colmap_unit=1.0, length_unit="m",
        world_frame=f"reference front camera: {aligned.reference_image}",
        front_frame=config.front_frame, front_name_pattern=config.front_name_pattern,
        front_camera_id=config.front_camera_id, alignment=alignment_info,
    )
    report_path = output_dir / "pipeline_report.json"
    report = {
        "status": "complete", "input_model": str(input_model),
        "aligned_model": str(aligned.output_model), "distance_file": str(distance_file),
        "extrinsics_file": str(extrinsics_file), "reference_image": aligned.reference_image,
        "front_frame": config.front_frame, "length_unit": "m",
        "world_axes": "Reference front-camera optical axes: X right, Y down, Z forward",
        "original_meters_per_colmap_unit": aligned.meters_per_input_unit,
        "aligned_model_meters_per_unit": 1.0,
        "frame_count": poses["metadata"]["frame_count"],
        "generated_side_pose_count": poses["metadata"]["generated_pose_count"],
        "exported_four_camera_pose_count": poses["metadata"]["exported_pose_count"],
        "poses_json": str(poses_dir / "poses.json"), "poses_csv": str(poses_dir / "poses.csv"),
        "alignment": alignment_info, "scale_fit": aligned.scale_report,
        "calibration": poses["metadata"]["calibration"], "warnings": poses["metadata"]["warnings"],
        "laser": None if laser_result is None else {
            "scan_file": str(scan_file), "ply_path": str(laser_result.ply_path),
            "poses_path": str(laser_result.poses_path), "report_path": str(laser_result.report_path),
            "scan_count": laser_result.scan_count, "used_scan_count": laser_result.used_scan_count,
            "point_count": laser_result.point_count, "excluded_pose_names": list(laser_result.excluded_pose_names),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    return PipelineResult(
        aligned.output_model, poses_dir / "poses.json", poses_dir / "poses.csv", report_path,
        aligned.reference_image, poses["metadata"]["frame_count"], aligned.meters_per_input_unit,
        laser_ply=None if laser_result is None else laser_result.ply_path,
        laser_poses=None if laser_result is None else laser_result.poses_path,
        laser_report=None if laser_result is None else laser_result.report_path,
    )
