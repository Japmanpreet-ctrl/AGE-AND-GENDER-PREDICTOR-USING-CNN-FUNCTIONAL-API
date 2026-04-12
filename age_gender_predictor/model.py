from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)

from age_gender_predictor.config import BACKBONE_WEIGHTS, IMAGE_SIZE


AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


AUGMENTATION = build_augmentation()


def load_training_image(path, age, gender):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    image = AUGMENTATION(image)
    return image, {"age": tf.cast(age, tf.float32), "gender": tf.cast(gender, tf.float32)}


def load_validation_image(path, age, gender):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, {"age": tf.cast(age, tf.float32), "gender": tf.cast(gender, tf.float32)}


def make_tf_dataset(frame, batch_size: int, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (frame["path"].values, frame["age"].values, frame["gender"].values)
    )
    mapper = load_training_image if training else load_validation_image
    dataset = dataset.map(mapper, num_parallel_calls=AUTOTUNE)
    if training:
        dataset = dataset.shuffle(1000)
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def build_model() -> tuple[Model, Model]:
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    weights = BACKBONE_WEIGHTS if BACKBONE_WEIGHTS not in {"", "none", "random"} else None
    base_model = ResNet50(weights=weights, include_top=False, input_tensor=inputs)
    for layer in base_model.layers:
        layer.trainable = False

    shared = base_model(inputs)

    age_branch = Conv2D(256, (3, 3), padding="same", activation="relu")(shared)
    age_branch = BatchNormalization()(age_branch)
    age_branch = GlobalAveragePooling2D()(age_branch)
    age_branch = Dense(128, activation="relu")(age_branch)
    age_branch = Dropout(0.3)(age_branch)
    age_branch = Dense(32, activation="relu")(age_branch)
    age_branch = Dropout(0.3)(age_branch)
    age_output = Dense(1, activation="relu", name="age")(age_branch)

    gender_branch = Conv2D(128, (3, 3), padding="same", activation="relu")(shared)
    gender_branch = BatchNormalization()(gender_branch)
    gender_branch = GlobalAveragePooling2D()(gender_branch)
    gender_branch = Dense(64, activation="relu")(gender_branch)
    gender_branch = Dropout(0.3)(gender_branch)
    gender_branch = Dense(16, activation="relu")(gender_branch)
    gender_branch = Dropout(0.3)(gender_branch)
    gender_output = Dense(1, activation="sigmoid", name="gender")(gender_branch)

    model = Model(inputs=inputs, outputs=[age_output, gender_output], name="age_gender_resnet50")
    return model, base_model


def compile_stage_one(model: Model) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={"age": tf.keras.losses.Huber(delta=5.0), "gender": "binary_crossentropy"},
        loss_weights={"age": 1.0, "gender": 0.7},
        metrics={"age": "mae", "gender": ["accuracy", tf.keras.metrics.AUC(name="auc")]},
    )


def compile_stage_two(model: Model, base_model: Model) -> None:
    for layer in base_model.layers:
        layer.trainable = layer.name.startswith("conv5")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss={"age": tf.keras.losses.Huber(delta=3.0), "gender": "binary_crossentropy"},
        loss_weights={"age": 1.0, "gender": 0.7},
        metrics={"age": "mae", "gender": ["accuracy", tf.keras.metrics.AUC(name="auc")]},
    )
