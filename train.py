"""
Training script for the music genre classification CNN.

Implements file-level stratified train/test split to prevent data leakage.

Usage:
    python train.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)

import config
from model import build_model


def load_data(spectrogram_dir=config.SPECTROGRAM_DIR):
    """Load preprocessed spectrograms, labels, and file map."""
    X = np.load(os.path.join(spectrogram_dir, "X.npy"))
    y = np.load(os.path.join(spectrogram_dir, "y.npy"))
    file_map = pd.read_csv(os.path.join(spectrogram_dir, "file_map.csv"))
    return X, y, file_map


def split_by_file(X, y, file_map, test_size=config.TEST_SPLIT,
                  seed=config.RANDOM_SEED):
    """
    Split data by original file to prevent data leakage.

    Chunks from the same song never appear in both train and test sets.

    Returns:
        X_train, X_test, y_train (one-hot), y_test (one-hot),
        test_file_map (DataFrame for majority voting evaluation)
    """
    # Get unique files with their genre
    unique_files = file_map[["genre", "original_file"]].drop_duplicates()

    # Stratified split on file level
    train_files, test_files = train_test_split(
        unique_files,
        test_size=test_size,
        random_state=seed,
        stratify=unique_files["genre"],
    )

    # Map back to chunk indices
    train_set = set(zip(train_files["genre"], train_files["original_file"]))
    test_set = set(zip(test_files["genre"], test_files["original_file"]))

    train_mask = file_map.apply(
        lambda r: (r["genre"], r["original_file"]) in train_set, axis=1
    )
    test_mask = file_map.apply(
        lambda r: (r["genre"], r["original_file"]) in test_set, axis=1
    )

    train_indices = file_map[train_mask].index.values
    test_indices = file_map[test_mask].index.values

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = to_categorical(y[train_indices], config.NUM_CLASSES)
    y_test = to_categorical(y[test_indices], config.NUM_CLASSES)

    test_file_map = file_map[test_mask].reset_index(drop=True)

    print(f"Train: {len(train_indices)} chunks from {len(train_files)} files")
    print(f"Test:  {len(test_indices)} chunks from {len(test_files)} files")

    return X_train, X_test, y_train, y_test, test_file_map


def train(X_train, y_train, X_val, y_val):
    """
    Build, train, and save the CNN model.

    Returns the trained model.
    """
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            config.MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
    )

    # Save training history
    history_dict = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"\nModel saved to: {config.MODEL_PATH}")
    print(f"History saved to: {config.HISTORY_PATH}")

    return model


if __name__ == "__main__":
    print("Loading preprocessed data...")
    X, y, file_map = load_data()

    print("Splitting dataset (file-level stratified)...")
    X_train, X_test, y_train, y_test, test_file_map = split_by_file(X, y, file_map)

    # Save test data for evaluation
    os.makedirs(config.SPECTROGRAM_DIR, exist_ok=True)
    np.save(os.path.join(config.SPECTROGRAM_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(config.SPECTROGRAM_DIR, "y_test.npy"), y_test)
    test_file_map.to_csv(
        os.path.join(config.SPECTROGRAM_DIR, "test_file_map.csv"), index=False
    )

    print("\nStarting training...")
    model = train(X_train, y_train, X_test, y_test)
