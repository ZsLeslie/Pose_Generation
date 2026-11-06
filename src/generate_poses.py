"""Propagate COLMAP front-camera poses through a calibrated camera rig.

All matrices use column vectors: X_A = H_A_from_B @ X_B.
The output world frame and length unit remain those of the input COLMAP model.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("left", "mid", "right")
CSV_CAMERAS = ("front", *CAMERAS)


@dataclass
class ImagePose:
    image_id: int
    camera_id: int
    name: str
    camera_from_world: np.ndarray


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    """Hamilton quaternion in COLMAP order (w, x, y, z)."""
    qvec = np.asarray(qvec, dtype=float)
    norm = np.linalg.norm(qvec)
    if qvec.shape != (4,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = qvec / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Use the largest component to remain stable near 180-degree rotations."""
    r = rotation
    squared = np.array([
        1 + r[0, 0] + r[1, 1] + r[2, 2],
        1 + r[0, 0] - r[1, 1] - r[2, 2],
        1 - r[0, 0] + r[1, 1] - r[2, 2],
        1 - r[0, 0] - r[1, 1] + r[2, 2],
    ])
    index = int(np.argmax(squared))
    component = 0.5 * math.sqrt(max(0.0, squared[index]))
    divisor = 4 * component
    if index == 0:
        qvec = np.array([component, (r[2, 1]-r[1, 2])/divisor,
                         (r[0, 2]-r[2, 0])/divisor, (r[1, 0]-r[0, 1])/divisor])
    elif index == 1:
        qvec = np.array([(r[2, 1]-r[1, 2])/divisor, component,
                         (r[0, 1]+r[1, 0])/divisor, (r[0, 2]+r[2, 0])/divisor])
    elif index == 2:
        qvec = np.array([(r[0, 2]-r[2, 0])/divisor,
                         (r[0, 1]+r[1, 0])/divisor, component,
                         (r[1, 2]+r[2, 1])/divisor])
    else:
        qvec = np.array([(r[1, 0]-r[0, 1])/divisor,
                         (r[0, 2]+r[2, 0])/divisor,
                         (r[1, 2]+r[2, 1])/divisor, component])
    qvec /= np.linalg.norm(qvec)
    return -qvec if qvec[0] < 0 else qvec


def validate_transform(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: expected a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"{name}: expected a rigid transform with det(R)=+1")
    return matrix


def invert_pose(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
    return inverse


def make_image_pose(image_id, camera_id, name, qvec, tvec) -> ImagePose:
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_rotation(qvec)
    matrix[:3, 3] = tvec
    validate_transform(matrix, f"Image {image_id}")
    if image_id <= 0 or camera_id <= 0 or not name:
        raise ValueError("Image IDs, camera IDs and image names must be nonempty/positive")
    return ImagePose(image_id, camera_id, name, matrix)


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated COLMAP file at byte {stream.tell()}")
    return data


def read_images_binary(path: Path) -> list[ImagePose]:
    """Read poses only; skip the 24-byte (x, y, point3D_id) observations."""
    images = []
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        count, = struct.unpack("<Q", read_exact(stream, 8))
        if count > max(0, file_size - 8) // 73:
            raise ValueError("Invalid COLMAP image count or truncated file")
        for _ in range(count):
            fields = struct.unpack("<i7di", read_exact(stream, 64))
            name_bytes = bytearray()
            while True:
                value = read_exact(stream, 1)
                if value == b"\0":
                    break
                name_bytes.extend(value)
            name = name_bytes.decode("utf-8")
            point_count, = struct.unpack("<Q", read_exact(stream, 8))
            skip_bytes = 24 * point_count
            if skip_bytes > file_size - stream.tell():
                raise ValueError(f"Truncated COLMAP observations for image {fields[0]}")
            stream.seek(skip_bytes, 1)
            images.append(make_image_pose(fields[0], fields[8], name,
                                          fields[1:5], fields[5:8]))
        if stream.read(1):
            raise ValueError("Unexpected trailing bytes in COLMAP images.bin")
    return images


def read_images_text(path: Path) -> list[ImagePose]:
    images = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.strip().split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError(f"Invalid COLMAP image header: {line.rstrip()}")
            images.append(make_image_pose(
                int(fields[0]), int(fields[8]), fields[9],
                list(map(float, fields[1:5])), list(map(float, fields[5:8])),
            ))
            # An empty POINTS2D line is valid and must still be consumed.
            observations = stream.readline()
            if observations == "" or len(observations.split()) % 3:
                raise ValueError(f"Missing/invalid POINTS2D line for image {fields[0]}")
    return images


def read_front_poses(model: Path, name_pattern: str, camera_id: int | None):
    if (model / "images.bin").is_file():
        source = model / "images.bin"
        images = read_images_binary(source)
    elif (model / "images.txt").is_file():
        source = model / "images.txt"
        images = read_images_text(source)
    else:
        raise FileNotFoundError(f"No images.bin or images.txt in {model}")
    if len({image.image_id for image in images}) != len(images):
        raise ValueError("Duplicate COLMAP image IDs")
    if len({image.name for image in images}) != len(images):
        raise ValueError("Duplicate COLMAP image names")
    images = [image for image in images
              if fnmatch.fnmatchcase(PurePosixPath(image.name.replace('\\', '/')).name,
                                    name_pattern)
              and (camera_id is None or image.camera_id == camera_id)]
    if not images:
        raise ValueError("No front images match the name pattern/camera ID")
    # The supplied names contain fixed-width timestamps; sorting preserves capture order.
    return sorted(images, key=lambda image: (image.name, image.image_id)), source


def load_extrinsics(path: Path, front_frame: str = "work"):
    if front_frame not in ("work", "capture"):
        raise ValueError("front_frame must be work or capture")
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
        # These H_*_from_front matrices ALREADY refer to front_work in this YAML.
        # For capture-frame COLMAP poses, undo the work conversion exactly once.
        if front_frame == "capture":
            work_from_capture = read_matrix("H_front_work_from_front_capture")
            transforms = {name: matrix @ work_from_capture
                          for name, matrix in transforms.items()}
        approximate_node = storage.getNode("extrinsics_approximate")
        metadata = {
            "translation_unit": unit,
            "reference_camera": reference,
            "front_pose": storage.getNode("front_pose").string(),
            "extrinsics_approximate": (None if approximate_node.empty()
                                       else bool(approximate_node.real())),
            "accuracy_note": storage.getNode("accuracy_note").string(),
        }
        return transforms, metadata
    finally:
        storage.release()


def propagate_pose(front_from_world: np.ndarray, camera_from_front_m: np.ndarray,
                   meters_per_colmap_unit: float) -> np.ndarray:
    if not math.isfinite(meters_per_colmap_unit) or meters_per_colmap_unit <= 0:
        raise ValueError("meters_per_colmap_unit must be finite and positive")
    camera_from_front = camera_from_front_m.copy()
    # Keep the COLMAP world/point cloud unchanged; convert rig meters to its units.
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


def generate_poses(model: Path, extrinsics: Path, output: Path, *,
                   meters_per_colmap_unit: float | None = None,
                   front_frame: str = "work", front_name_pattern: str = "front_*",
                   front_camera_id: int | None = None) -> dict:
    scale = 1.0 if meters_per_colmap_unit is None else meters_per_colmap_unit
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("meters_per_colmap_unit must be finite and positive")
    transforms, calibration = load_extrinsics(extrinsics, front_frame)
    images, source = read_front_poses(model, front_name_pattern, front_camera_id)
    warnings = []
    if meters_per_colmap_unit is None:
        warnings.append("COLMAP scale is unknown: provisionally assuming 1 unit = 1 m. "
                        "Camera positions are provisional; rotations do not depend on scale.")
    if calibration["extrinsics_approximate"]:
        warnings.append("The YAML declares approximate extrinsics: " + calibration["accuracy_note"])
    for warning in warnings:
        print("WARNING: " + warning, file=sys.stderr)
    metadata = {
        "source_images": str(source.resolve()),
        "extrinsics_file": str(extrinsics.resolve()),
        "front_frame": front_frame,
        "front_name_pattern": front_name_pattern,
        "front_camera_id_filter": front_camera_id,
        "frame_count": len(images),
        "generated_pose_count": len(images) * len(CAMERAS),
        "world_frame": "unchanged COLMAP world frame",
        "length_unit": "colmap_unit",
        "meters_per_colmap_unit": scale,
        "scale_status": "assumed" if meters_per_colmap_unit is None else "provided",
        "quaternion_order": "wxyz",
        "quaternion_direction": "world_to_camera",
        "matrix_definition": "X_A = H_A_from_B @ X_B; column vectors",
        "calibration": calibration,
        "warnings": warnings,
    }
    frames = []
    for image in images:
        poses = {"front": pose_record(image.camera_from_world)}
        for camera, transform in transforms.items():
            poses[camera] = pose_record(propagate_pose(image.camera_from_world, transform, scale))
        frames.append({
            "source_image_id": image.image_id,
            "source_camera_id": image.camera_id,
            "front_image_name": image.name,
            "poses": poses,
        })
    result = {"metadata": metadata, "frames": frames}
    output.mkdir(parents=True, exist_ok=True)
    (output / "poses.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "poses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "source_image_id", "source_camera_id", "front_image_name", "camera",
            "qw", "qx", "qy", "qz", "tx", "ty", "tz", "cx", "cy", "cz",
            "meters_per_colmap_unit", "scale_status", "front_frame",
        ])
        for frame in frames:
            for camera in CSV_CAMERAS:
                pose = frame["poses"][camera]
                writer.writerow([
                    frame["source_image_id"], frame["source_camera_id"],
                    frame["front_image_name"], camera,
                    *pose["qvec_wxyz"], *pose["tvec"], *pose["camera_center_world"],
                    scale, metadata["scale_status"], front_frame,
                ])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "colmap/sparse")
    parser.add_argument("--extrinsics", type=Path,
                        default=PROJECT_ROOT / "extrinsics/rig_extrinsics.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--meters-per-colmap-unit", type=float, default=None,
                        help="Meters in one COLMAP unit; omission uses a PROVISIONAL value of 1")
    parser.add_argument("--front-frame", choices=("work", "capture"), default="work",
                        help="Front optical frame used by COLMAP (default: work)")
    parser.add_argument("--front-name-pattern", default="front_*",
                        help="Case-sensitive glob matched against image basenames")
    parser.add_argument("--front-camera-id", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        result = generate_poses(
            args.model, args.extrinsics, args.output,
            meters_per_colmap_unit=args.meters_per_colmap_unit,
            front_frame=args.front_frame, front_name_pattern=args.front_name_pattern,
            front_camera_id=args.front_camera_id,
        )
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(2, f"ERROR: {error}\n")
    metadata = result["metadata"]
    print(f"Read {metadata['frame_count']} front poses; generated "
          f"{metadata['generated_pose_count']} left/mid/right poses.")
    print(f"Scale: {metadata['meters_per_colmap_unit']:g} m/COLMAP unit "
          f"({metadata['scale_status']}); front frame: {args.front_frame}")
    print(f"Saved: {args.output.resolve() / 'poses.json'}")
    print(f"Saved: {args.output.resolve() / 'poses.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
