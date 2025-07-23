# src/train.py

import keras_tuner as kt
import tensorflow as tf

from config import *
from data_preprocessing import download_data, load_and_preprocess_data
from model import WineQualityHyperModel
import os

def run_tuning_and_training():
    """Main function to run hyperparameter tuning and final training."""
    # Ensure data is available
    if not os.path.exists(DATA_PATH):
        download_data(DATA_URL, DATA_PATH, DATA_SEPARATOR)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, num_classes = load_and_preprocess_data(
        DATA_PATH, DATA_SEPARATOR, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
    )
    
    input_shape = (X_train.shape[1],)
    hypermodel = WineQualityHyperModel(input_shape=input_shape, num_classes=num_classes)
    
    # --- KerasTuner Hyperparameter Search ---
    tuner = kt.RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='keras_tuner_dir',
        project_name=KERAS_TUNER_PROJECT_NAME
    )
    
    print("\n--- Starting Hyperparameter Search ---")
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
    
    # --- Get and Evaluate Best Model ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    
    print("\n--- Training Best Model on Full Training Data ---")
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[stop_early])

    print("\n--- Best Model Summary ---")
    model.summary()
    print(f"""
    The hyperparameter search is complete. 
    Optimal number of layers: {best_hps.get('num_layers')}
    Optimal learning rate: {best_hps.get('lr'):.4f}
    """)
    
    print("\n--- Evaluating Best Model on Test Data ---")
    eval_result = model.evaluate(X_test, y_test)
    print(f"[test loss, test accuracy]: [{eval_result[0]:.4f}, {eval_result[1]:.4f}]")
    
    # --- Save the Model ---
    model.save(MODEL_SAVE_PATH)
    print(f"\nBest model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    run_tuning_and_training()