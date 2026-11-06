"""Read cumulative travel distances from TXT or convert Excel to TXT first."""

import math
from pathlib import Path


EXCEL_DISTANCE_SUFFIXES = {".xlsx", ".xlsm", ".xls", ".excel"}
DISTANCE_HEADERS = {"distance", "distance_m", "cumulative_distance", "距离", "累计距离"}
IMAGE_NAME_HEADERS = {"name", "image", "image_name", "filename", "file_name", "图像名称", "图片名称"}


def iter_excel_distance_rows(path):
    """Read the first worksheet; select XLS/XLSX reader by file contents."""
    path = Path(path)
    with path.open("rb") as stream:
        signature = stream.read(8)
        stream.seek(0)
        if signature == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            try:
                import xlrd
            except ImportError as error:
                raise RuntimeError("Reading XLS requires: python -m pip install xlrd") from error
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
                raise RuntimeError("Reading Excel requires: python -m pip install openpyxl") from error
            # A binary stream permits valid XLSX workbooks renamed to .excel.
            workbook = load_workbook(stream, read_only=True, data_only=True)
            try:
                yield from enumerate(workbook.worksheets[0].iter_rows(values_only=True), start=1)
            finally:
                workbook.close()


def convert_distance_excel_to_txt(path):
    """Export validated 'distance image_name' records, preserving existing TXT data."""
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
                    raise ValueError(f"{location}: expected name/distance headers or two data columns") from error
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
    path = Path(path)
    if path.suffix.lower() in EXCEL_DISTANCE_SUFFIXES:
        text_path = convert_distance_excel_to_txt(path)
        print(f"Distance Excel converted to TXT: {path} -> {text_path}")
        path = text_path
    distances = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
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
