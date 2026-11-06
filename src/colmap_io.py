"""Read registered image poses without a COLMAP runtime dependency."""

from __future__ import annotations

import fnmatch
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import numpy as np

from .geometry import camera_center, quaternion_to_rotation, validate_transform


@dataclass
class ImagePose:
    image_id: int
    camera_id: int
    name: str
    camera_from_world: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return camera_center(self.camera_from_world[:3, :3], self.camera_from_world[:3, 3])


def natural_key(name: str):
    return tuple((0, int(part)) if part.isdigit() else (1, part)
                 for part in re.split(r"(\d+)", name))


def make_image_pose(image_id, camera_id, name, qvec, tvec) -> ImagePose:
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_rotation(qvec)
    matrix[:3, 3] = tvec
    validate_transform(matrix, f"Image {image_id}")
    if image_id <= 0 or camera_id <= 0 or not name:
        raise ValueError("Image IDs, camera IDs and names must be nonempty/positive")
    return ImagePose(image_id, camera_id, name, matrix)


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated COLMAP file at byte {stream.tell()}")
    return data


def read_images_binary(path: Path) -> list[ImagePose]:
    images = []
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        count, = struct.unpack("<Q", read_exact(stream, 8))
        if count > max(0, file_size - 8) // 73:
            raise ValueError("Invalid COLMAP image count or truncated file")
        for _ in range(count):
            fields = struct.unpack("<i7di", read_exact(stream, 64))
            name_bytes = bytearray()
            while (value := read_exact(stream, 1)) != b"\0":
                name_bytes.extend(value)
            point_count, = struct.unpack("<Q", read_exact(stream, 8))
            skip_bytes = 24 * point_count
            if skip_bytes > file_size - stream.tell():
                raise ValueError(f"Truncated COLMAP observations for image {fields[0]}")
            stream.seek(skip_bytes, 1)
            images.append(make_image_pose(fields[0], fields[8], name_bytes.decode("utf-8"),
                                          fields[1:5], fields[5:8]))
        if stream.read(1):
            raise ValueError("Unexpected trailing bytes in COLMAP images.bin")
    return images


def read_images_text(path: Path) -> list[ImagePose]:
    images = []
    with Path(path).open(encoding="utf-8-sig") as stream:
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
            observations = stream.readline()
            if observations == "" or len(observations.split()) % 3:
                raise ValueError(f"Missing/invalid POINTS2D line for image {fields[0]}")
    return images


def read_front_poses(model: Path, name_pattern="front_*", camera_id=None):
    model = Path(model)
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
              if fnmatch.fnmatchcase(PurePosixPath(image.name.replace('\\', '/')).name, name_pattern)
              and (camera_id is None or image.camera_id == camera_id)]
    if not images:
        raise ValueError("No front images match the name pattern/camera ID")
    return sorted(images, key=lambda image: (natural_key(image.name), image.image_id)), source
