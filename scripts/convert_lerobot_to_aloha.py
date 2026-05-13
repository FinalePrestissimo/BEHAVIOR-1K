"""
Convert BEHAVIOR-1K LeRobot episodes into an Aloha-style raw HDF5 layout.

The source dataset is the R1Pro-based `datasets/2025-challenge-demos` release.
This script reads each episode parquet together with its RGB videos and writes
an HDF5 file with the standard Aloha top-level layout:

- /observations/qpos
- /observations/qvel
- /observations/effort
- /observations/images/{cam_high, cam_low, cam_left_wrist, cam_right_wrist}
- /action
- /relative_action
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import av
import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_SOURCE_ROOT = Path("datasets/2025-challenge-demos")
TARGET_IMAGE_SIZE = (480, 480)  # width, height
IMAGE_SHAPE = (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], 3)

SOURCE_VIDEO_KEYS = {
    "cam_high": "observation.images.rgb.head",
    # "cam_low": "observation.images.rgb.head",
    "cam_left_wrist": "observation.images.rgb.left_wrist",
    "cam_right_wrist": "observation.images.rgb.right_wrist",
}


def find_parquet_files(source_root: Path) -> list[Path]:
    return sorted(source_root.glob("data/task-*/episode_*.parquet"))


def resolve_video_path(source_root: Path, parquet_path: Path, video_key: str) -> Path:
    task_dir = parquet_path.parent.name
    return source_root / "videos" / task_dir / video_key / f"{parquet_path.stem}.mp4"


def load_episode_arrays(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path)

    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)

    if state.ndim != 2 or state.shape[1] != 256:
        raise ValueError(f"Expected observation.state to have shape (T, 256), got {state.shape} in {parquet_path}")
    if action.ndim != 2 or action.shape[1] != 23:
        raise ValueError(f"Expected action to have shape (T, 23), got {action.shape} in {parquet_path}")

    return state, action


def project_r1pro_proprio(state: np.ndarray) -> np.ndarray:
    """
    refer to b1k-baselines/baselines/openvla-oft/RLDS_builder/behavior_dataset/utils/data_utils.py
    """

    base_qvel = state[:,253:256] # 3
    trunk_qpos = state[:,236:240] # 4
    arm_left_qpos = state[:,158:165] #  7
    arm_right_qpos = state[:,197:204] #  7
    left_gripper_width = state[:,193:195].sum(axis=-1)[:,None] # 1
    right_gripper_width = state[:,232:234].sum(axis=-1)[:,None] # 1
    
    prop_state = np.concatenate((base_qvel, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width), axis=-1) # 23
    
    return prop_state.astype(np.float32)


def project_r1pro_action(action: np.ndarray) -> np.ndarray:
    if action.ndim != 2 or action.shape[1] != 23:
        raise ValueError(f"Expected a R1Pro action vector with shape (T, 23), got {action.shape}")

    base = action[:, 0:3]
    torso = action[:, 3:7]
    left_arm = action[:, 7:14]
    left_gripper = action[:, 14:15]
    right_arm = action[:, 15:22]
    right_gripper = action[:, 22:23]

    return np.concatenate(
        [base, torso, left_arm, left_gripper, right_arm, right_gripper],
        axis=1,
    ).astype(np.float32)


def compute_relative_action(action: np.ndarray) -> np.ndarray:
    relative_action = np.zeros_like(action)
    if len(action) > 1:
        relative_action[:-1] = action[1:] - action[:-1]
        relative_action[-1] = np.zeros(action.shape[1], dtype=action.dtype)
    return relative_action


def iter_resized_frames(video_path: Path, image_size: tuple[int, int]) -> Iterator[np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            yield cv2.resize(rgb, image_size, interpolation=cv2.INTER_AREA)


def stream_video_to_targets(
    video_path: Path,
    target_datasets: dict[str, h5py.Dataset],
    image_size: tuple[int, int],
    expected_frames: int,
) -> None:
    written_frames = 0
    for frame in iter_resized_frames(video_path, image_size):
        for dataset in target_datasets.values():
            dataset[written_frames] = frame
        written_frames += 1

    if written_frames != expected_frames:
        raise ValueError(
            f"Video length mismatch for {video_path}: wrote {written_frames} frames, expected {expected_frames}"
        )


def write_episode_hdf5(source_root: Path, parquet_path: Path, output_root: Path, overwrite: bool) -> Path:
    task_dir = parquet_path.parent.name
    output_path = output_root / task_dir / f"{parquet_path.stem}.hdf5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"Skipping existing file {output_path} (use --overwrite to overwrite)")
        return output_path


    # see OmniGibson/omnigibson/learning/utils/eval_utils.py for detailed order
    state, action = load_episode_arrays(parquet_path)
    qpos = project_r1pro_proprio(state)
    action_23 = project_r1pro_action(action)
    relative_action = compute_relative_action(action_23)

    num_frames = qpos.shape[0]
    if action_23.shape[0] != num_frames:
        raise ValueError(f"Frame count mismatch in {parquet_path}: qpos={num_frames}, action={action_23.shape[0]}")

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["sim"] = False

        observations = h5f.create_group("observations")
        observations.create_dataset("qpos", data=qpos, dtype="float32")

        images = observations.create_group("images")
        image_datasets: dict[str, h5py.Dataset] = {}
        for camera_name in SOURCE_VIDEO_KEYS:
            image_datasets[camera_name] = images.create_dataset(
                camera_name,
                shape=(num_frames, *IMAGE_SHAPE),
                dtype="uint8",
                chunks=(1, *IMAGE_SHAPE),
                compression="lzf",
            )


        h5f.create_dataset("action", data=action_23, dtype="float32")
        h5f.create_dataset("relative_action", data=relative_action, dtype="float32")

        source_to_targets: dict[str, list[str]] = {}
        for target_camera, source_video_key in SOURCE_VIDEO_KEYS.items():
            source_to_targets.setdefault(source_video_key, []).append(target_camera)

        for source_video_key, target_cameras in source_to_targets.items():
            video_path = resolve_video_path(source_root, parquet_path, source_video_key)
            stream_targets = {camera_name: image_datasets[camera_name] for camera_name in target_cameras}
            stream_video_to_targets(video_path, stream_targets, TARGET_IMAGE_SIZE, num_frames)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LeRobot BEHAVIOR-1K episodes to Aloha HDF5 files.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root of the LeRobot dataset (defaults to datasets/2025-challenge-demos)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where Aloha HDF5 files are written",
    )
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N episodes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    parquet_files = find_parquet_files(args.source_root)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.source_root / 'data'}")
    if args.limit is not None:
        parquet_files = parquet_files[: args.limit]


    args.output_root.mkdir(parents=True, exist_ok=True)
    for parquet_path in tqdm(parquet_files, desc="Converting episodes"):
        write_episode_hdf5(args.source_root, parquet_path, args.output_root, args.overwrite)
