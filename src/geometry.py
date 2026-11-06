"""Shared pose math: column vectors, Hamilton quaternions in (w, x, y, z)."""

import math

import numpy as np


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
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
    """Use the largest component for stability near half-turn rotations."""
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


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return -rotation.T @ translation


def camera_forward_world(rotation: np.ndarray) -> np.ndarray:
    return rotation.T[:, 2].copy()


def invert_pose(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = camera_center(matrix[:3, :3], matrix[:3, 3])
    return inverse
