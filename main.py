"""Run metric COLMAP alignment, laser scan stitching and camera rig pose generation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from src.pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=defaults.input_model,
                        help="Sparse model directory containing images.bin or images.txt")
    parser.add_argument("--distance", type=Path, default=defaults.distance_file,
                        help="Cumulative distance readings in meters: TXT or Excel")
    parser.add_argument("--extrinsics", type=Path, default=defaults.extrinsics_file,
                        help="OpenCV rig extrinsics YAML")
    parser.add_argument("--output", type=Path, default=defaults.output_dir,
                        help="Pipeline output directory")
    parser.add_argument("--reference-image", help="Reference front image; default: first registered front image")
    parser.add_argument("--colmap-bin", default=defaults.colmap_bin, help="COLMAP executable path or command")
    parser.add_argument("--front-frame", choices=("work", "capture"), default=defaults.front_frame)
    parser.add_argument("--front-name-pattern", default=defaults.front_name_pattern)
    parser.add_argument("--front-camera-id", type=int)
    parser.add_argument("--scan", type=Path, default=defaults.scan_file,
                        help="LaserScan JSON Lines file; default: scan/scan.json")
    parser.add_argument("--skip-laser", action="store_true", help="Run camera alignment and rig poses only")
    parser.add_argument("--laser-extrinsics", type=Path,
                        help="Optional OpenCV YAML containing H_front_from_laser and translation_unit: m")
    parser.add_argument("--laser-distance-step", type=float, default=defaults.laser_distance_step_m,
                        help="Laser distance grid in meters; 0 uses all unique scan distances")
    parser.add_argument("--keep-laser-pose-outliers", action="store_true",
                        help="Disable pose-outlier exclusion for laser interpolation")
    parser.add_argument("--sync-pose-outlier-filter", action="store_true",
                        help="Also exclude laser-detected outlier frames from four-camera pose exports")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing pipeline results")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PipelineConfig(
        input_model=args.input, distance_file=args.distance, extrinsics_file=args.extrinsics,
        output_dir=args.output, reference_image=args.reference_image, colmap_bin=args.colmap_bin,
        front_frame=args.front_frame, front_name_pattern=args.front_name_pattern,
        front_camera_id=args.front_camera_id, overwrite=args.overwrite,
        scan_file=None if args.skip_laser else args.scan, laser_extrinsics_file=args.laser_extrinsics,
        laser_distance_step_m=args.laser_distance_step,
        laser_filter_pose_outliers=not args.keep_laser_pose_outliers,
        sync_pose_outlier_filter=args.sync_pose_outlier_filter,
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = run_pipeline(config)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Completed: {result.frame_count} front frames and {result.frame_count * 3} side poses")
    print(f"Reference world frame: {result.reference_image}; output unit: meters")
    print(f"Aligned model: {result.aligned_model}")
    print(f"Four-camera poses: {result.poses_csv}")
    if config.sync_pose_outlier_filter:
        print(f"Camera frames excluded using laser pose-outlier detection: {len(result.excluded_pose_names)}")
    if result.laser_ply is not None:
        print(f"Stitched laser cloud: {result.laser_ply}")
        print(f"Laser report: {result.laser_report}")
    print(f"Report: {result.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
