from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.getenv("UTKFACE_DATASET_DIR", PROJECT_ROOT / "data" / "UTKFace"))
MODEL_PATH = Path(os.getenv("AGE_GENDER_MODEL_PATH", PROJECT_ROOT / "models" / "age_gender_resnet50.keras"))
MODEL_INFO_PATH = Path(os.getenv("AGE_GENDER_MODEL_INFO_PATH", PROJECT_ROOT / "models" / "age_gender_resnet50.json"))
IMAGE_SIZE = int(os.getenv("AGE_GENDER_IMAGE_SIZE", "224"))
BATCH_SIZE = int(os.getenv("AGE_GENDER_BATCH_SIZE", "32"))
TRAIN_HEAD_EPOCHS = int(os.getenv("AGE_GENDER_TRAIN_HEAD_EPOCHS", "3"))
TRAIN_FINETUNE_EPOCHS = int(os.getenv("AGE_GENDER_TRAIN_FINETUNE_EPOCHS", "5"))
DATASET_LIMIT = int(os.getenv("AGE_GENDER_DATASET_LIMIT", "0"))
BACKBONE_WEIGHTS = os.getenv("AGE_GENDER_BACKBONE_WEIGHTS", "imagenet").strip().lower()
RANDOM_STATE = 42
TRAIN_SPLIT = 0.8
AGE_OUTLIER_CUTOFF = 80
GENDER_UNCERTAINTY_LOW = 0.45
GENDER_UNCERTAINTY_HIGH = 0.55
APP_TITLE = "Age & Gender Predictor"
APP_HOST = os.getenv("AGE_GENDER_APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("AGE_GENDER_APP_PORT", "7860"))
APP_SHARE = os.getenv("AGE_GENDER_APP_SHARE", "false").strip().lower() in {"1", "true", "yes", "on"}
