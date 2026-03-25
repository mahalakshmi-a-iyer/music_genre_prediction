"""
Single-file genre prediction with majority voting.

Usage:
    python inference.py <path_to_wav_file>
"""

import sys
import numpy as np
from collections import Counter
from tensorflow import keras

import config
from preprocess import preprocess_single_file


def load_model(model_path=config.MODEL_PATH):
    """Load the trained Keras model."""
    return keras.models.load_model(model_path)


def predict_genre(file_path, model=None, model_path=config.MODEL_PATH):
    """
    Predict the genre of a .wav file using majority voting across chunks.

    Args:
        file_path: Path to the .wav file.
        model: Pre-loaded Keras model (optional, avoids reloading).
        model_path: Path to the .keras model file.

    Returns:
        dict with keys:
            - predicted_genre: str
            - confidence: float (fraction of chunks voting for the winner)
            - chunk_predictions: list[str]
            - genre_probabilities: dict[str, float] (averaged across chunks)
    """
    if model is None:
        model = load_model(model_path)

    # Preprocess: chunk and extract spectrograms
    X = preprocess_single_file(file_path)

    # Predict all chunks
    predictions = model.predict(X, verbose=0)  # shape: (num_chunks, 10)

    # Per-chunk predicted genres
    chunk_pred_indices = np.argmax(predictions, axis=1)
    chunk_pred_genres = [config.GENRES[i] for i in chunk_pred_indices]

    # Majority vote
    vote_counts = Counter(chunk_pred_indices)
    winner_idx, winner_count = vote_counts.most_common(1)[0]
    confidence = winner_count / len(chunk_pred_indices)

    # Average probabilities across chunks
    avg_probs = predictions.mean(axis=0)
    genre_probabilities = {
        genre: float(prob) for genre, prob in zip(config.GENRES, avg_probs)
    }

    return {
        "predicted_genre": config.GENRES[winner_idx],
        "confidence": float(confidence),
        "chunk_predictions": chunk_pred_genres,
        "genre_probabilities": genre_probabilities,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_wav_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"Predicting genre for: {file_path}")

    result = predict_genre(file_path)

    print(f"\nPredicted Genre: {result['predicted_genre']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\nChunk predictions: {result['chunk_predictions']}")
    print(f"\nGenre probabilities:")
    for genre, prob in sorted(
        result["genre_probabilities"].items(), key=lambda x: -x[1]
    ):
        print(f"  {genre:12s}: {prob:.4f}")
