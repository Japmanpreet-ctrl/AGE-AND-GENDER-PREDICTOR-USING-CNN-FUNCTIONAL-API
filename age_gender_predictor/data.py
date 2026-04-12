from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from age_gender_predictor.config import AGE_OUTLIER_CUTOFF, DATASET_DIR, DATASET_LIMIT, RANDOM_STATE, TRAIN_SPLIT


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


def _parse_filename(path: Path) -> dict | None:
    parts = path.name.split("_")
    if len(parts) < 4:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
    except ValueError:
        return None

    return {
        "path": str(path),
        "age": age,
        "gender": gender,
    }


def build_dataframe(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at {dataset_dir}. Set UTKFACE_DATASET_DIR to the UTKFace image folder."
        )

    rows = []
    image_paths = sorted(dataset_dir.rglob("*.jpg"))
    if DATASET_LIMIT > 0:
        image_paths = image_paths[:DATASET_LIMIT]

    for image_path in image_paths:
        parsed = _parse_filename(image_path)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        raise FileNotFoundError(
            f"No valid .jpg files were found in {dataset_dir}. Point UTKFACE_DATASET_DIR at the aligned UTKFace folder."
        )

    frame = pd.DataFrame(rows)
    frame = frame[frame["age"] < AGE_OUTLIER_CUTOFF].reset_index(drop=True)
    return frame


def make_splits(frame: pd.DataFrame) -> DatasetSplit:
    train_df, validation_df = train_test_split(
        frame,
        test_size=1 - TRAIN_SPLIT,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return DatasetSplit(train=train_df.reset_index(drop=True), validation=validation_df.reset_index(drop=True))
