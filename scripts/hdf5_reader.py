"""Inspect and visualize the contents of an HDF5 file.

The script prints the internal group/dataset tree, shows compact summaries for
datasets, and can export image-like datasets to a preview directory.

Examples:
    python scripts/hdf5_reader.py path/to/file.hdf5
    python scripts/hdf5_reader.py path/to/file.hdf5 --preview
    python scripts/hdf5_reader.py path/to/file.hdf5 --preview --dataset /observations/images/cam_high
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency for previews only.
    cv2 = None


DEFAULT_MAX_INLINE_VALUES = 8
DEFAULT_PREVIEW_LIMIT = 6
DEFAULT_PREVIEW_COLUMNS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the HDF5 tree and optionally export image previews.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5_path", type=Path, help="Path to the HDF5 file to inspect")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum recursion depth for the tree view; omit for full depth",
    )
    parser.add_argument(
        "--max-inline-values",
        type=int,
        default=DEFAULT_MAX_INLINE_VALUES,
        help="How many values to show inline for small numeric datasets",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Export previews for image-like datasets",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Directory for exported preview images",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=DEFAULT_PREVIEW_LIMIT,
        help="Maximum number of image-like datasets to preview; 0 means no limit",
    )
    parser.add_argument(
        "--preview-frames",
        type=int,
        default=4,
        help="Maximum number of frames to include in each image preview grid",
    )
    parser.add_argument(
        "--preview-columns",
        type=int,
        default=DEFAULT_PREVIEW_COLUMNS,
        help="Number of columns to use in preview contact sheets",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Internal HDF5 dataset path to preview; can be repeated",
    )
    return parser.parse_args()


def format_scalar(value: object, max_length: int = 96) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            value = value.decode("utf-8", errors="replace")
    text = repr(value)
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def format_array_preview(array: np.ndarray, max_length: int = 160) -> str:
    text = np.array2string(array, separator=", ", threshold=array.size, edgeitems=2, precision=4, suppress_small=True)
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def is_text_dataset(dataset: h5py.Dataset) -> bool:
    return dataset.dtype.kind in {"S", "O", "U"}


def is_numeric_dataset(dataset: h5py.Dataset) -> bool:
    return np.issubdtype(dataset.dtype, np.number)


def channel_axis_for_shape(shape: tuple[int, ...]) -> int | None:
    if len(shape) == 2:
        return None
    if len(shape) == 3:
        if shape[-1] in {1, 3, 4}:
            return -1
        if shape[0] in {1, 3, 4}:
            return 0
    if len(shape) == 4:
        if shape[-1] in {1, 3, 4}:
            return -1
        if shape[1] in {1, 3, 4}:
            return 1
    return None


def is_image_like_dataset(dataset: h5py.Dataset) -> bool:
    if not is_numeric_dataset(dataset):
        return False
    return channel_axis_for_shape(dataset.shape) is not None and max(dataset.shape, default=0) >= 8


def describe_attrs(attrs: h5py.AttributeManager, max_inline_values: int) -> str:
    if len(attrs) == 0:
        return ""

    pieces: list[str] = []
    for key in sorted(attrs.keys()):
        value = attrs[key]
        if isinstance(value, np.ndarray):
            if value.size <= max_inline_values:
                rendered = format_array_preview(value)
            else:
                rendered = f"ndarray shape={value.shape} dtype={value.dtype}"
        else:
            rendered = format_scalar(value)
        pieces.append(f"{key}={rendered}")
    return ", ".join(pieces)


def sample_numeric_dataset(dataset: h5py.Dataset, max_inline_values: int) -> np.ndarray | None:
    try:
        if dataset.ndim == 0:
            return np.asarray(dataset[()])

        if dataset.ndim == 1:
            limit = min(dataset.shape[0], max_inline_values)
            return np.asarray(dataset[:limit])

        if dataset.ndim == 2:
            rows = min(dataset.shape[0], 3)
            cols = min(dataset.shape[1], max_inline_values)
            return np.asarray(dataset[:rows, :cols])

        slices = tuple(slice(0, min(size, 2)) for size in dataset.shape)
        return np.asarray(dataset[slices])
    except Exception:
        return None


def dataset_summary(dataset: h5py.Dataset, max_inline_values: int) -> str:
    pieces = [f"shape={dataset.shape}", f"dtype={dataset.dtype}"]
    if dataset.compression:
        pieces.append(f"compression={dataset.compression}")
    if dataset.chunks:
        pieces.append(f"chunks={dataset.chunks}")
    if dataset.shuffle:
        pieces.append("shuffle=True")
    if dataset.fletcher32:
        pieces.append("fletcher32=True")

    if dataset.ndim == 0:
        pieces.append(f"value={format_scalar(dataset[()])}")
        return " ".join(pieces)

    if is_text_dataset(dataset):
        try:
            if dataset.size == 0:
                pieces.append("sample=[]")
            else:
                first_item = dataset[0]
                if isinstance(first_item, np.ndarray):
                    preview = format_array_preview(first_item)
                else:
                    preview = format_scalar(first_item)
                pieces.append(f"sample={preview}")
        except Exception:
            pieces.append("sample=<unavailable>")
        return " ".join(pieces)

    if is_numeric_dataset(dataset):
        sample = sample_numeric_dataset(dataset, max_inline_values)
        if sample is not None:
            pieces.append(f"sample={format_array_preview(sample)}")
        return " ".join(pieces)

    pieces.append(f"sample_type={dataset.dtype}")
    return " ".join(pieces)


def print_attrs(label: str, attrs: h5py.AttributeManager, indent: str, max_inline_values: int) -> None:
    rendered = describe_attrs(attrs, max_inline_values)
    if rendered:
        print(f"{indent}{label} attrs: {rendered}")


def collect_dataset_paths(group: h5py.Group) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, item: h5py.ObjectProxy) -> None:
        if isinstance(item, h5py.Dataset):
            paths.append("/" + name)

    group.visititems(visitor)
    return paths


def normalize_image_array(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)

    if data.ndim == 2:
        data = data[:, :, None]

    channel_axis = channel_axis_for_shape(data.shape)
    if channel_axis == 0:
        data = np.moveaxis(data, 0, -1)
    elif channel_axis == 1:
        data = np.moveaxis(data, 1, -1)

    if data.ndim != 3:
        raise ValueError(f"Expected an image-like array, got shape {data.shape}")

    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    elif data.shape[-1] > 4:
        raise ValueError(f"Unsupported channel count for preview: {data.shape[-1]}")

    if not np.issubdtype(data.dtype, np.uint8):
        if np.issubdtype(data.dtype, np.floating):
            max_value = float(np.nanmax(data)) if data.size else 0.0
            min_value = float(np.nanmin(data)) if data.size else 0.0
            if min_value >= 0.0 and max_value <= 1.5:
                data = np.clip(data, 0.0, 1.0) * 255.0
            else:
                data = np.clip(data, 0.0, 255.0)
            data = data.astype(np.uint8)
        else:
            data = np.clip(data, 0, 255).astype(np.uint8)

    return data


def make_contact_sheet(frames: list[np.ndarray], columns: int = 2, padding: int = 8) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot build a contact sheet from an empty frame list")

    normalized_frames = [normalize_image_array(frame) for frame in frames]
    tile_height, tile_width = normalized_frames[0].shape[:2]
    columns = max(1, columns)
    rows = math.ceil(len(normalized_frames) / columns)

    canvas = np.zeros(
        (
            rows * tile_height + (rows + 1) * padding,
            columns * tile_width + (columns + 1) * padding,
            3,
        ),
        dtype=np.uint8,
    )

    for index, frame in enumerate(normalized_frames):
        row = index // columns
        column = index % columns
        if frame.shape[:2] != (tile_height, tile_width):
            frame = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

        top = padding + row * (tile_height + padding)
        left = padding + column * (tile_width + padding)
        canvas[top : top + tile_height, left : left + tile_width] = frame
        label = f"{index}"
        cv2.putText(
            canvas,
            label,
            (left + 8, top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def decode_encoded_image(raw: object) -> np.ndarray | None:
    if cv2 is None:
        return None

    if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
        buffer = raw
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        buffer = np.frombuffer(raw, dtype=np.uint8)
    else:
        return None

    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def preview_dataset(
    dataset: h5py.Dataset,
    dataset_path: str,
    preview_dir: Path,
    preview_frames: int,
    preview_columns: int,
) -> Path | None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for image previews but is not installed")

    preview_dir.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_path.strip("/").replace("/", "__") or "root"
    output_path = preview_dir / f"{safe_name}.png"

    if dataset.ndim == 3:
        array = np.asarray(dataset[()])
        image = normalize_image_array(array)
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return output_path

    if dataset.ndim == 4:
        limit = min(dataset.shape[0], max(1, preview_frames))
        frames = [np.asarray(dataset[index]) for index in range(limit)]
        sheet = make_contact_sheet(frames, columns=preview_columns)
        cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        return output_path

    if is_text_dataset(dataset) and dataset.size > 0:
        first_item = dataset[0]
        decoded = decode_encoded_image(first_item)
        if decoded is not None:
            cv2.imwrite(str(output_path), cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR))
            return output_path

    return None


def walk_group(
    group: h5py.Group,
    prefix: str,
    current_depth: int,
    max_depth: int | None,
    max_inline_values: int,
    preview_paths: set[str],
    preview_dir: Path | None,
    preview_frames: int,
    preview_columns: int,
) -> None:
    names = sorted(group.keys())
    for index, name in enumerate(names):
        item = group[name]
        is_last = index == len(names) - 1
        branch = "`-- " if is_last else "|-- "
        child_prefix = "    " if is_last else "|   "

        if isinstance(item, h5py.Group):
            print(f"{prefix}{branch}{name}/")
            print_attrs("group", item.attrs, prefix + child_prefix, max_inline_values)
            if max_depth is None or current_depth < max_depth:
                walk_group(
                    item,
                    prefix + child_prefix,
                    current_depth + 1,
                    max_depth,
                    max_inline_values,
                    preview_paths,
                    preview_dir,
                    preview_frames,
                    preview_columns,
                )
            elif len(item.keys()) > 0:
                print(f"{prefix}{child_prefix}...")
        else:
            summary = dataset_summary(item, max_inline_values)
            print(f"{prefix}{branch}{name} [{summary}]")
            print_attrs("dataset", item.attrs, prefix + child_prefix, max_inline_values)

            full_path = item.name
            if preview_dir is not None and full_path in preview_paths:
                preview_path = preview_dataset(item, full_path, preview_dir, preview_frames, preview_columns)
                if preview_path is not None:
                    print(f"{prefix}{child_prefix}preview: {preview_path}")


def inspect_file(
    hdf5_path: Path,
    max_depth: int | None,
    max_inline_values: int,
    preview: bool,
    preview_dir: Path | None,
    preview_limit: int,
    preview_frames: int,
    preview_columns: int,
    explicit_preview_datasets: list[str],
) -> None:
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Not a file: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as root:
        print(f"HDF5 file: {hdf5_path}")
        print_attrs("file", root.attrs, "", max_inline_values)

        preview_paths: set[str] = set()
        if preview:
            if explicit_preview_datasets:
                preview_paths = {path if path.startswith("/") else f"/{path.lstrip('/')}" for path in explicit_preview_datasets}
            else:
                candidates = [path for path in collect_dataset_paths(root) if is_image_like_dataset(root[path])]
                if preview_limit > 0:
                    candidates = candidates[:preview_limit]
                preview_paths = set(candidates)

        walk_group(
            root,
            prefix="",
            current_depth=0,
            max_depth=max_depth,
            max_inline_values=max_inline_values,
            preview_paths=preview_paths,
            preview_dir=preview_dir if preview else None,
            preview_frames=preview_frames,
            preview_columns=preview_columns,
        )


def main() -> None:
    args = parse_args()

    preview_dir = args.preview_dir
    if args.preview and preview_dir is None:
        preview_dir = args.hdf5_path.with_name(f"{args.hdf5_path.stem}_preview")

    inspect_file(
        hdf5_path=args.hdf5_path,
        max_depth=args.max_depth,
        max_inline_values=args.max_inline_values,
        preview=args.preview,
        preview_dir=preview_dir,
        preview_limit=args.preview_limit,
        preview_frames=args.preview_frames,
        preview_columns=args.preview_columns,
        explicit_preview_datasets=args.datasets,
    )


if __name__ == "__main__":
    main()