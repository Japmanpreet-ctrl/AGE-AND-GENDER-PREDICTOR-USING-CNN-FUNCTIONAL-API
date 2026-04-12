from __future__ import annotations

import json
from pathlib import Path
import sys

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from age_gender_predictor.config import (
    BATCH_SIZE,
    BACKBONE_WEIGHTS,
    DATASET_DIR,
    DATASET_LIMIT,
    MODEL_PATH,
    MODEL_INFO_PATH,
    TRAIN_FINETUNE_EPOCHS,
    TRAIN_HEAD_EPOCHS,
)
from age_gender_predictor.data import build_dataframe, make_splits
from age_gender_predictor.model import (
    build_model,
    compile_stage_one,
    compile_stage_two,
    make_tf_dataset,
)


def main() -> None:
    frame = build_dataframe()
    print(f"Using dataset directory: {DATASET_DIR}")
    print(f"Found {len(frame)} usable images after filtering.")
    if DATASET_LIMIT > 0:
        print(f"Dataset limit enabled: {DATASET_LIMIT}")

    splits = make_splits(frame)

    train_ds = make_tf_dataset(splits.train, batch_size=BATCH_SIZE, training=True)
    validation_ds = make_tf_dataset(splits.validation, batch_size=BATCH_SIZE, training=False)

    model, base_model = build_model()
    compile_stage_one(model)
    stage_one_history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=TRAIN_HEAD_EPOCHS,
    )

    compile_stage_two(model, base_model)
    stage_two_history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=TRAIN_FINETUNE_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_gender_auc",
                patience=4,
                mode="max",
                restore_best_weights=True,
            )
        ],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    MODEL_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_training_manifest(
        model_path=MODEL_PATH,
        model_info_path=MODEL_INFO_PATH,
        stage_one_history=stage_one_history.history,
        stage_two_history=stage_two_history.history,
        dataset_size=len(frame),
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved model metadata to {MODEL_INFO_PATH}")


def save_training_manifest(
    model_path: Path,
    model_info_path: Path,
    stage_one_history: dict[str, list[float]],
    stage_two_history: dict[str, list[float]],
    dataset_size: int,
) -> None:
    final_metrics = {
        key: values[-1]
        for key, values in stage_two_history.items()
        if values
    }
    manifest = {
        "model_path": str(model_path),
        "dataset_dir": str(DATASET_DIR),
        "dataset_size": dataset_size,
        "dataset_limit": DATASET_LIMIT,
        "batch_size": BATCH_SIZE,
        "backbone_weights": BACKBONE_WEIGHTS,
        "train_head_epochs": TRAIN_HEAD_EPOCHS,
        "train_finetune_epochs": TRAIN_FINETUNE_EPOCHS,
        "stage_one_history": stage_one_history,
        "stage_two_history": stage_two_history,
        "final_metrics": final_metrics,
    }
    model_info_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
