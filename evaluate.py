"""
Model evaluation with chunk-level and file-level (majority voting) metrics.

Usage:
    python evaluate.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
)
from tensorflow import keras

import config


def plot_training_history(history_path=config.HISTORY_PATH,
                          save_dir=config.PLOTS_DIR):
    """Plot and save training/validation accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path, "r") as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history["accuracy"], label="Train Accuracy")
    ax1.plot(history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history["loss"], label="Train Loss")
    ax2.plot(history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")


def evaluate_chunks(model, X_test, y_test):
    """
    Evaluate model at the chunk level.

    Returns classification report as a dictionary.
    """
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\n" + "=" * 50)
    print("CHUNK-LEVEL EVALUATION")
    print("=" * 50)
    print(classification_report(
        y_true_classes, y_pred_classes, target_names=config.GENRES,
    ))

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    print(f"Chunk-level accuracy: {accuracy:.4f}")

    return classification_report(
        y_true_classes, y_pred_classes,
        target_names=config.GENRES, output_dict=True,
    )


def evaluate_majority_vote(model, X_test, y_test, test_file_map,
                           save_dir=config.PLOTS_DIR):
    """
    Evaluate model at the file level using majority voting.

    Groups chunks by original file, predicts each chunk, takes the mode.

    Returns classification report as a dictionary.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Group by original file and apply majority voting
    file_predictions = []
    file_true_labels = []

    grouped = test_file_map.groupby(["genre", "original_file"])

    for (genre, filename), group in grouped:
        indices = group.index.values
        chunk_preds = y_pred_classes[indices]

        # Majority vote
        vote_counts = Counter(chunk_preds)
        majority_genre_idx = vote_counts.most_common(1)[0][0]

        file_predictions.append(majority_genre_idx)
        file_true_labels.append(config.GENRES.index(genre))

    file_predictions = np.array(file_predictions)
    file_true_labels = np.array(file_true_labels)

    print("\n" + "=" * 50)
    print("FILE-LEVEL EVALUATION (MAJORITY VOTING)")
    print("=" * 50)
    print(classification_report(
        file_true_labels, file_predictions, target_names=config.GENRES,
    ))

    accuracy = accuracy_score(file_true_labels, file_predictions)
    print(f"File-level accuracy: {accuracy:.4f}")

    # Confusion matrix
    plot_confusion_matrix(
        file_true_labels, file_predictions, config.GENRES,
        os.path.join(save_dir, "confusion_matrix.png"),
    )

    return classification_report(
        file_true_labels, file_predictions,
        target_names=config.GENRES, output_dict=True,
    )


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.title("Confusion Matrix (File-Level, Majority Voting)")
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def load_or_create_test_data():
    """Load cached test data, or create it from the full dataset."""
    x_test_path = os.path.join(config.SPECTROGRAM_DIR, "X_test.npy")
    y_test_path = os.path.join(config.SPECTROGRAM_DIR, "y_test.npy")
    map_path = os.path.join(config.SPECTROGRAM_DIR, "test_file_map.csv")

    if os.path.exists(x_test_path) and os.path.exists(y_test_path) and os.path.exists(map_path):
        print("Loading cached test data...")
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        test_file_map = pd.read_csv(map_path)
    else:
        print("Test data not found. Generating from full dataset...")
        from train import load_data, split_by_file
        X, y, file_map = load_data()
        _, X_test, _, y_test, test_file_map = split_by_file(X, y, file_map)

        np.save(x_test_path, X_test)
        np.save(y_test_path, y_test)
        test_file_map.to_csv(map_path, index=False)
        print("Test data saved for future runs.")

    return X_test, y_test, test_file_map


if __name__ == "__main__":
    print("Loading model and test data...")
    model = keras.models.load_model(config.MODEL_PATH)

    X_test, y_test, test_file_map = load_or_create_test_data()

    # Plot training history
    if os.path.exists(config.HISTORY_PATH):
        plot_training_history()
    else:
        print("Training history not found. Skipping training curves plot.")

    # Chunk-level evaluation
    evaluate_chunks(model, X_test, y_test)

    # File-level evaluation with majority voting
    evaluate_majority_vote(model, X_test, y_test, test_file_map)
