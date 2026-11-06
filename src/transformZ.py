#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transformZ.py

把指定 COLMAP 参考相机定义为新的世界坐标系：

    reference camera center = (0, 0, 0)
    reference camera +X     = world +X
    reference camera +Y     = world +Y
    reference camera +Z     = world +Z

因此：
    参考帧相机光轴 -> 世界 +Z
    根据 distance TXT / Excel 文件的累计行进距离（米）恢复整个模型的统一尺度

不依赖 pycolmap。
使用 COLMAP 自带：

    model_converter
    model_transformer

输入：
    sparse/0

输出：
    sparse_first_camera/

同时保存：
    T_new_from_old.txt
    transform_sim3.txt
    scale_report.json

满足：

    X_new = scale * (R_ref @ X_old + t_ref)
"""

import argparse
import shutil
import math
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# =============================================================================
# Utilities
# =============================================================================

def run_command(cmd):
    print()
    print("$ " + " ".join(str(x) for x in cmd))
    print()

    result = subprocess.run(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}:\n"
            + " ".join(str(x) for x in cmd)
        )


def qvec_to_rotmat(q):
    """
    COLMAP Hamilton quaternion:

        q = [qw, qx, qy, qz]

    world -> camera rotation.
    """

    q = np.asarray(q, dtype=np.float64)

    norm = np.linalg.norm(q)

    if norm < 1e-15:
        raise ValueError("Invalid zero quaternion.")

    q = q / norm

    qw, qx, qy, qz = q

    R = np.array([
        [
            1.0 - 2.0 * (qy*qy + qz*qz),
            2.0 * (qx*qy - qz*qw),
            2.0 * (qx*qz + qy*qw),
        ],
        [
            2.0 * (qx*qy + qz*qw),
            1.0 - 2.0 * (qx*qx + qz*qz),
            2.0 * (qy*qz - qx*qw),
        ],
        [
            2.0 * (qx*qz - qy*qw),
            2.0 * (qy*qz + qx*qw),
            1.0 - 2.0 * (qx*qx + qy*qy),
        ],
    ], dtype=np.float64)

    return R


def camera_center(R, t):
    """
    COLMAP:
        X_cam = R * X_world + t

    Camera center:
        C = -R^T t
    """
    return -R.T @ t


def camera_forward_world(R):
    """
    相机坐标系 +Z 光轴在世界坐标中的方向：

        z_world = R^T [0,0,1]
    """

    z_cam = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float64
    )

    z_world = R.T @ z_cam

    n = np.linalg.norm(z_world)

    if n > 0:
        z_world /= n

    return z_world


# =============================================================================
# Read COLMAP images.txt
# =============================================================================

def read_images_txt(images_txt):
    """
    读取 COLMAP images.txt。

    返回：
        dict[name] = {
            "image_id": ...,
            "qvec": ...,
            "tvec": ...,
            "camera_id": ...
        }

    COLMAP images.txt 第一行格式：

        IMAGE_ID
        QW QX QY QZ
        TX TY TZ
        CAMERA_ID
        NAME

    第二行为 POINTS2D。
    """

    images = {}

    with open(images_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # comment / blank
        if not line or line.startswith("#"):
            i += 1
            continue

        tokens = line.split()

        # Image pose line 至少 10 项
        if len(tokens) < 10:
            i += 1
            continue

        try:
            image_id = int(tokens[0])

            qw = float(tokens[1])
            qx = float(tokens[2])
            qy = float(tokens[3])
            qz = float(tokens[4])

            tx = float(tokens[5])
            ty = float(tokens[6])
            tz = float(tokens[7])

            camera_id = int(tokens[8])

        except ValueError:
            # 大概率是 POINTS2D 行
            i += 1
            continue

        # IMAGE NAME 通常不含空格，但这里兼容
        name = " ".join(tokens[9:])

        images[name] = {
            "image_id": image_id,
            "qvec": np.array(
                [qw, qx, qy, qz],
                dtype=np.float64
            ),
            "tvec": np.array(
                [tx, ty, tz],
                dtype=np.float64
            ),
            "camera_id": camera_id,
        }

        # images.txt 每张图占两行。
        # 后面一行是 POINTS2D。
        i += 2

    return images


def find_reference(images, requested_name):

    # exact match
    if requested_name in images:
        return requested_name, images[requested_name]

    # basename match
    requested_base = Path(requested_name).name

    candidates = [
        (name, data)
        for name, data in images.items()
        if Path(name).name == requested_base
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple images match basename '{requested_base}':\n"
            + "\n".join(
                "    " + x[0]
                for x in candidates
            )
        )

    print()
    print("Reference image not found.")
    print()
    print("Some available image names:")

    for name in list(images.keys())[:30]:
        print("    " + name)

    raise RuntimeError(
        f"Reference image not found: {requested_name}"
    )


# =============================================================================
# Metric scale from cumulative travel distances
# =============================================================================


EXCEL_DISTANCE_SUFFIXES = {".xlsx", ".xlsm", ".xls", ".excel"}
DISTANCE_HEADERS = {"distance", "distance_m", "cumulative_distance", "距离", "累计距离"}
IMAGE_NAME_HEADERS = {"name", "image", "image_name", "filename", "file_name", "图像名称", "图片名称"}


def iter_excel_distance_rows(path):
    """Read the first worksheet; select XLS/XLSX reader by file contents."""
    with path.open("rb") as stream:
        signature = stream.read(8)
        stream.seek(0)
        if signature == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            try:
                import xlrd
            except ImportError as error:
                raise RuntimeError(
                    "Reading .xls distance files requires xlrd: "
                    "python -m pip install xlrd"
                ) from error
            workbook = xlrd.open_workbook(str(path), on_demand=True)
            try:
                sheet = workbook.sheet_by_index(0)
                for index in range(sheet.nrows):
                    yield index + 1, sheet.row_values(index)
            finally:
                workbook.release_resources()
        else:
            try:
                from openpyxl import load_workbook
            except ImportError as error:
                raise RuntimeError(
                    "Reading Excel distance files requires openpyxl: "
                    "python -m pip install openpyxl"
                ) from error
            # Passing a stream also supports a valid workbook renamed to .excel.
            workbook = load_workbook(stream, read_only=True, data_only=True)
            try:
                sheet = workbook.worksheets[0]
                yield from enumerate(sheet.iter_rows(values_only=True), start=1)
            finally:
                workbook.close()


def convert_distance_excel_to_txt(path):
    """Export validated records as 'distance image_name', without a header."""
    path = Path(path)
    columns = None
    records = {}
    for row_no, row in iter_excel_distance_rows(path):
        if all(value is None or str(value).strip() == "" for value in row):
            continue
        location = f"{path}:row {row_no}"
        if columns is None:
            headers = [str(value).strip().lower() if value is not None else "" for value in row]
            distance_columns = [i for i, value in enumerate(headers) if value in DISTANCE_HEADERS]
            name_columns = [i for i, value in enumerate(headers) if value in IMAGE_NAME_HEADERS]
            if distance_columns or name_columns:
                if len(distance_columns) != 1 or len(name_columns) != 1:
                    raise ValueError(f"{location}: expected one distance column and one name column")
                columns = distance_columns[0], name_columns[0]
                continue
            # Also accept a headerless table: distance/name or name/distance.
            if len(row) < 2:
                raise ValueError(f"{location}: expected distance and image name columns")
            try:
                float(row[0])
                columns = 0, 1
            except (ValueError, TypeError):
                try:
                    float(row[1])
                    columns = 1, 0
                except (ValueError, TypeError) as error:
                    raise ValueError(
                        f"{location}: expected name/distance headers or a headerless two-column table"
                    ) from error
        distance_column, name_column = columns
        if len(row) <= max(columns):
            raise ValueError(f"{location}: missing distance or image name")
        value, name = row[distance_column], row[name_column]
        try:
            distance = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{location}: invalid or missing distance {value!r}") from error
        if isinstance(value, bool) or not math.isfinite(distance) or distance < 0:
            raise ValueError(f"{location}: distance must be finite and >= 0")
        if not isinstance(name, str) or not name.strip() or "\n" in name or "\r" in name:
            raise ValueError(f"{location}: missing or invalid image name")
        name = name.strip().replace("\\", "/")
        if name in records:
            raise ValueError(f"{location}: duplicate image {name}")
        records[name] = distance
    if not records:
        raise ValueError(f"No distance records in {path}")
    text = "".join(f"{distance:.17g} {name}\n" for name, distance in records.items())
    # Reuse an identical export; preserve any existing TXT with different data.
    index = 0
    while True:
        suffix = "" if index == 0 else ".converted" + (str(index) if index > 1 else "")
        target = path.with_name(f"{path.stem}{suffix}.txt")
        try:
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            return target
        except FileExistsError:
            try:
                if target.is_file() and target.read_text(encoding="utf-8-sig") == text:
                    return target
            except (OSError, UnicodeError):
                pass
            index += 1


def read_distances(path):
    """Convert Excel inputs to TXT, then read cumulative meters/image names."""
    path = Path(path)
    if path.suffix.lower() in EXCEL_DISTANCE_SUFFIXES:
        text_path = convert_distance_excel_to_txt(path)
        print(f"Distance Excel converted to TXT: {path} -> {text_path}")
        path = text_path
    distances = {}
    for line_no, line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: expected distance image_name")
        value, name = parts
        distance = float(value)
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"{path}:{line_no}: distance must be finite and >= 0")
        name = name.replace("\\", "/")
        if name in distances:
            raise ValueError(f"{path}:{line_no}: duplicate image {name}")
        distances[name] = distance
    if not distances:
        raise ValueError(f"No distance records in {path}")
    return distances


def estimate_metric_scale(images, distances):
    """Fit s * cumulative camera-path length to cumulative meter increments.

    The distance values determine travel order. All registered cameras must
    have a distance. Unregistered distance records are ignored because their
    camera centers are unknown. Camera-to-camera chords approximate the path.
    The first matched reading is subtracted; it need not be zero.
    """
    by_basename = {}
    for name, distance in distances.items():
        by_basename.setdefault(name.rsplit("/", 1)[-1], []).append((name, distance))
    matched = []
    used = set()
    for name, pose in images.items():
        normalized = name.replace("\\", "/")
        if normalized in distances:
            distance_name, distance = normalized, distances[normalized]
        else:
            candidates = by_basename.get(normalized.rsplit("/", 1)[-1], [])
            if len(candidates) != 1:
                raise ValueError(f"Missing or ambiguous distance for image: {name}")
            distance_name, distance = candidates[0]
        if distance_name in used:
            raise ValueError(f"Distance record matches multiple cameras: {distance_name}")
        used.add(distance_name)
        center = camera_center(qvec_to_rotmat(pose["qvec"]), pose["tvec"])
        if not np.all(np.isfinite(center)):
            raise ValueError(f"Non-finite camera center: {name}")
        matched.append((distance, name, center))
    if len(matched) < 2:
        raise ValueError("At least two registered cameras with distances are required")
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
    report = {
        "scale_meters_per_model_unit": scale,
        "matched_images": len(matched),
        "unused_distance_records": len(distances) - len(used),
        "first_image": matched[0][1],
        "last_image": matched[-1][1],
        "first_distance_m": float(readings[0]),
        "last_distance_m": float(readings[-1]),
        "model_path_length": float(model_path[-1]),
        "distance_span_m": float(meter_path[-1]),
        "fit_rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
        "fit_max_abs_error_m": float(np.max(np.abs(residuals))),
    }
    return scale, report


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input COLMAP sparse model, e.g. sparse/0",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output COLMAP sparse model",
    )

    parser.add_argument(
        "--reference-image",
        required=True,
        type=str,
        help="Reference image name",
    )

    parser.add_argument(
        "--distance",
        required=True,
        type=Path,
        help=("Distance TXT or Excel (.xlsx/.xls/.xlsm/.excel). "
              "Excel is converted to TXT first; use name/distance column headers."),
    )

    parser.add_argument(
        "--colmap-bin",
        default="colmap",
        type=str,
        help="COLMAP executable. Default: colmap",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )

    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    distances = read_distances(args.distance)

    # =========================================================================
    # Check COLMAP
    # =========================================================================

    colmap_path = shutil.which(args.colmap_bin)

    if colmap_path is None:
        raise RuntimeError(
            f"COLMAP executable not found: {args.colmap_bin}\n"
            "Check:\n"
            "    which colmap"
        )

    print("=" * 80)
    print("COLMAP CAMERA-COORDINATE ALIGNMENT")
    print("=" * 80)

    print(f"COLMAP    : {colmap_path}")
    print(f"Input     : {input_dir}")
    print(f"Output    : {output_dir}")
    print(f"Reference : {args.reference_image}")

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    if input_dir == output_dir or output_dir in input_dir.parents:
        raise RuntimeError("Output must not equal or contain the input directory.")
    if output_dir in args.distance.resolve().parents:
        raise RuntimeError("Distance file must be outside the output directory.")

    # =========================================================================
    # Temporary text conversion
    # =========================================================================

    with tempfile.TemporaryDirectory(
        prefix="colmap_align_"
    ) as temp_root:

        temp_root = Path(temp_root)

        text_before = temp_root / "before"
        text_after = temp_root / "after"

        text_before.mkdir()
        text_after.mkdir()

        print()
        print("=" * 80)
        print("1. CONVERT ORIGINAL MODEL TO TEXT")
        print("=" * 80)

        run_command([
            colmap_path,
            "model_converter",
            "--input_path",
            input_dir,
            "--output_path",
            text_before,
            "--output_type",
            "TXT",
        ])

        images_txt = text_before / "images.txt"

        if not images_txt.exists():
            raise RuntimeError(
                f"COLMAP did not generate:\n{images_txt}"
            )

        # =====================================================================
        # Load reference pose
        # =====================================================================

        images = read_images_txt(images_txt)

        print(f"Registered images parsed: {len(images)}")

        ref_name, ref = find_reference(
            images,
            args.reference_image,
        )

        q_ref = ref["qvec"]
        t_ref = ref["tvec"]

        R_ref = qvec_to_rotmat(q_ref)

        C_ref = camera_center(
            R_ref,
            t_ref
        )

        forward_ref = camera_forward_world(
            R_ref
        )

        scale, scale_report = estimate_metric_scale(images, distances)
        print("\nMETRIC SCALE FIT")
        print(json.dumps(scale_report, indent=2, ensure_ascii=False))
        print("Scale uses camera-path chords between registered frames.")

        # =========================================================================
        # Output directory
        # =========================================================================

        if output_dir.exists():

            if any(output_dir.iterdir()):

                if not args.overwrite:
                    raise RuntimeError(
                        f"Output directory already exists and is not empty:\n"
                        f"{output_dir}\n\n"
                        "Use --overwrite if you want to replace it."
                    )

                shutil.rmtree(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        print()
        print("=" * 80)
        print("2. REFERENCE CAMERA ORIGINAL POSE")
        print("=" * 80)

        print()
        print(f"Image name : {ref_name}")
        print(f"Image ID   : {ref['image_id']}")
        print()

        print("qvec [qw qx qy qz]:")
        print(q_ref)

        print()
        print("tvec:")
        print(t_ref)

        print()
        print("R_world_to_camera:")
        print(R_ref)

        print()
        print("Camera center in OLD world:")
        print(C_ref)

        print()
        print("Camera optical axis +Z in OLD world:")
        print(forward_ref)

        # =====================================================================
        # Compute transform
        #
        # Reference camera:
        #
        #   X_cam = R_ref X_old + t_ref
        #
        # Define new world:
        #
        #   X_new = scale * (R_ref X_old + t_ref)
        #
        # Therefore reference camera becomes:
        #
        #   X_cam_new = X_new (camera-space lengths are scaled too)
        #
        # i.e.
        #
        #   R_new = I
        #   t_new = 0
        #
        # =====================================================================

        q_transform = q_ref.astype(np.float64).copy()

        # Make sure quaternion is unit length.
        q_transform /= np.linalg.norm(q_transform)
        
        t_transform = scale * t_ref

        transform_path = (
            output_dir /
            "transform_sim3.txt"
        )
        
        with open(transform_path, "w") as f:
            f.write(
                "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n"
                % (
                    scale,
                    q_transform[0],   # qw
                    q_transform[1],   # qx
                    q_transform[2],   # qy
                    q_transform[3],   # qz
                    t_transform[0],   # tx
                    t_transform[1],   # ty
                    t_transform[2],   # tz
                )
            )
        
        
        # 4x4 matrix is still useful for YOUR later coordinate conversions,
        # but it must NOT be passed to COLMAP model_transformer.
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = scale * R_ref
        T[:3, 3] = t_transform
        
        matrix_path = (
            output_dir /
            "T_new_from_old.txt"
        )
        
        np.savetxt(
            matrix_path,
            T,
            fmt="%.17g"
        )
        
        
        print()
        print("COLMAP Sim3 transform file:")
        print(f"    {transform_path}")
        
        print()
        print("4x4 coordinate transformation matrix:")
        print(f"    {matrix_path}")
        
        print()
        print("COLMAP Sim3:")
        print(
            f"    scale={scale:.12f} "
            f"q=[{q_transform[0]:.12f}, "
            f"{q_transform[1]:.12f}, "
            f"{q_transform[2]:.12f}, "
            f"{q_transform[3]:.12f}] "
            f"t=[{t_transform[0]:.12f}, "
            f"{t_transform[1]:.12f}, "
            f"{t_transform[2]:.12f}]"
        )

        print()
        print("=" * 80)
        print("3. TRANSFORMATION")
        print("=" * 80)

        print()
        print("T_new_from_old:")
        print(T)

        print()
        print(
            "Coordinate convention:"
        )
        print(
            "    X_new = scale * (R_ref @ X_old + t_ref)"
        )

        # =====================================================================
        # COLMAP model_transformer
        # =====================================================================

        print()
        print("=" * 80)
        print("4. TRANSFORM COMPLETE COLMAP MODEL")
        print("=" * 80)

        run_command([
            colmap_path,
            "model_transformer",
            "--input_path",
            input_dir,
            "--output_path",
            output_dir,
            "--transform_path",
            transform_path,
            "--is_inverse",
            "0",
        ])

        (output_dir / "scale_report.json").write_text(
            json.dumps(scale_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # =====================================================================
        # Verify
        # =====================================================================

        print()
        print("=" * 80)
        print("5. VERIFY TRANSFORMED MODEL")
        print("=" * 80)

        run_command([
            colmap_path,
            "model_converter",
            "--input_path",
            output_dir,
            "--output_path",
            text_after,
            "--output_type",
            "TXT",
        ])

        images_after = read_images_txt(
            text_after / "images.txt"
        )

        ref_name_after, ref_after = find_reference(
            images_after,
            ref_name,
        )

        verified_scale, _ = estimate_metric_scale(images_after, distances)
        print(f"Output model scale check (expected 1): {verified_scale:.12f}")
        if not np.isclose(verified_scale, 1.0, rtol=1e-7, atol=1e-8):
            raise RuntimeError("Transformed model failed metric scale verification")

        q_after = ref_after["qvec"]
        t_after = ref_after["tvec"]

        R_after = qvec_to_rotmat(
            q_after
        )

        C_after = camera_center(
            R_after,
            t_after
        )

        forward_after = camera_forward_world(
            R_after
        )

        print()
        print(f"Reference image: {ref_name_after}")

        print()
        print("q_after:")
        print(q_after)

        print()
        print("R_after:")
        print(R_after)

        print()
        print("t_after:")
        print(t_after)

        print()
        print("Camera center after:")
        print(C_after)

        print()
        print("Optical axis after:")
        print(forward_after)

        rot_error = np.linalg.norm(
            R_after - np.eye(3)
        )

        t_error = np.linalg.norm(
            t_after
        )

        c_error = np.linalg.norm(
            C_after
        )

        z_error = np.linalg.norm(
            forward_after -
            np.array([0.0, 0.0, 1.0])
        )

        print()
        print("-" * 80)

        print(
            f"||R-I||             = {rot_error:.12e}"
        )

        print(
            f"||t||               = {t_error:.12e}"
        )

        print(
            f"||C||               = {c_error:.12e}"
        )

        print(
            f"||Z-[0,0,1]||       = {z_error:.12e}"
        )

        tolerance = 1e-8

        print()

        if (
            rot_error < tolerance
            and
            t_error < tolerance
            and
            c_error < tolerance
            and
            z_error < tolerance
        ):
            print(
                "ALIGNMENT CHECK: OK"
            )
        else:
            print(
                "WARNING: alignment verification error "
                "is larger than expected."
            )

    # =========================================================================
    # Done
    # =========================================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print(f"Output model:")
    print(f"    {output_dir}")

    print()
    print(f"Transform matrix:")
    print(f"    {transform_path}")

    print()
    print("New coordinate convention:")
    print()
    print("    origin = reference camera center")
    print("    +X     = reference image right")
    print("    +Y     = reference image down")
    print("    +Z     = reference camera optical axis")
    print("    units  = meters (estimated from cumulative travel distances)")
    print()


if __name__ == "__main__":
    main()
