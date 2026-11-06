"""Run metric COLMAP alignment followed by left/mid/right rig pose generation."""

from __future__ import annotations

import argparse
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
    )
    try:
        result = run_pipeline(config)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Completed: {result.frame_count} front frames and {result.frame_count * 3} side poses")
    print(f"Reference world frame: {result.reference_image}; output unit: meters")
    print(f"Aligned model: {result.aligned_model}")
    print(f"Four-camera poses: {result.poses_csv}")
    print(f"Report: {result.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
