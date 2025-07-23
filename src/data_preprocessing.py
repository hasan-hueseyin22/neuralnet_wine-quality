# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests
import os

def download_data(url, save_path, separator):
    """Downloads data from a URL and saves it locally."""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    response = requests.get(url)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Data downloaded and saved to {save_path}")

def load_and_preprocess_data(path, separator, target_column, test_size, random_state):
    """Loads, preprocesses, and splits the data."""
    df = pd.read_csv(path, sep=separator)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Encode labels to start from 0
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get number of unique classes for the output layer
    num_classes = len(label_encoder.classes_)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, num_classes