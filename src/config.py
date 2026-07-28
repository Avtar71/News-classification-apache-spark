from pathlib import Path

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"
IMAGE_DIR = ROOT_DIR / "images"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# --------------------------------------------------
# Spark Configuration
# --------------------------------------------------

APP_NAME = "NewsClassification"

MASTER = "local[2]"

DRIVER_MEMORY = "4g"

SHUFFLE_PARTITIONS = "8"

LOG_LEVEL = "WARN"

# --------------------------------------------------
# Dataset Configuration
# --------------------------------------------------

LABEL_COLUMN = "label"

TITLE_COLUMN = "title"

DESCRIPTION_COLUMN = "description"

RAW_TEXT_COLUMN = "raw_text"

CLEAN_TEXT_COLUMN = "clean_text"

WORDS_COLUMN = "words"

FILTERED_WORDS_COLUMN = "filtered_words"

RAW_FEATURES_COLUMN = "raw_features"

FEATURES_COLUMN = "features"

PREDICTION_COLUMN = "prediction"

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

NUM_FEATURES = 2 ** 15

MIN_DOC_FREQ = 3

# --------------------------------------------------
# Training
# --------------------------------------------------

RANDOM_STATE = 42

MAX_ITER = 100

REG_PARAM = 0.01

NUM_TREES = 100

MAX_DEPTH = 10

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

METRICS = [
    "accuracy",
    "f1"
]
