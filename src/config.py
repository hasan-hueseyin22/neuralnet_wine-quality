# Paths
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
DATA_PATH = "data/raw/winequality-white.csv"
MODEL_SAVE_PATH = "models/best_wine_quality_model"
KERAS_TUNER_PROJECT_NAME = "wine_quality_tuning"

# Data specifics
TARGET_COLUMN = "quality"
# Delimiter for the CSV file
DATA_SEPARATOR = ";"

# Model & Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# KerasTuner settings
MAX_TRIALS = 20  # Number of hyperparameter combinations to test
EXECUTION_PER_TRIAL = 2 # Number of models to train and evaluate for each trial
