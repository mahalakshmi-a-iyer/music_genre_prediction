"""
Audio preprocessing pipeline: chunking, Mel spectrogram extraction, and dataset preparation.

Usage:
    python preprocess.py
"""

import os
import numpy as np
import pandas as pd
import librosa
from PIL import Image

import config


def load_and_chunk(file_path, sr=config.SAMPLE_RATE,
                   chunk_samples=config.CHUNK_SAMPLES,
                   overlap_samples=config.OVERLAP_SAMPLES):
    """
    Load a .wav file and split it into overlapping chunks.

    Returns a list of 1-D numpy arrays (raw audio chunks).
    For a 30s file: ~13 chunks of 4s each with 2s overlap.
    """
    audio, _ = librosa.load(file_path, sr=sr, res_type="kaiser_fast")

    step = chunk_samples - overlap_samples
    chunks = []

    for start in range(0, len(audio) - chunk_samples + 1, step):
        chunk = audio[start : start + chunk_samples]
        chunks.append(chunk)

    return chunks


def audio_to_mel_spectrogram(audio_chunk, sr=config.SAMPLE_RATE,
                              n_mels=config.N_MELS,
                              hop_length=config.HOP_LENGTH,
                              n_fft=config.N_FFT):
    """
    Compute Mel spectrogram from an audio chunk and convert to dB scale.

    Returns a 2-D numpy array of shape (n_mels, time_frames).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio_chunk, sr=sr, n_mels=n_mels,
        hop_length=hop_length, n_fft=n_fft,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def resize_spectrogram(spec, height=config.IMG_HEIGHT, width=config.IMG_WIDTH):
    """
    Resize a spectrogram to fixed dimensions and normalize to [0, 1].

    Returns a float32 array of shape (height, width).
    """
    img = Image.fromarray(spec)
    img = img.resize((width, height), Image.LANCZOS)
    resized = np.array(img, dtype=np.float64)

    # Min-max normalize to [0, 1]
    spec_min = resized.min()
    spec_max = resized.max()
    if spec_max - spec_min > 0:
        resized = (resized - spec_min) / (spec_max - spec_min)
    else:
        resized = np.zeros_like(resized)

    return resized.astype(np.float32)


def preprocess_dataset(data_dir=config.DATA_DIR, output_dir=config.SPECTROGRAM_DIR):
    """
    Process the entire GTZAN dataset: chunk, extract spectrograms, save as .npy.

    Saves:
        - X.npy: shape (N, 150, 150, 1), float32
        - y.npy: shape (N,), int32 labels 0-9
        - file_map.csv: chunk_index, genre, original_file, chunk_number

    Returns (X, y) arrays.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_spectrograms = []
    all_labels = []
    file_map_rows = []
    chunk_index = 0
    skipped_files = []

    for genre_idx, genre in enumerate(config.GENRES):
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_dir):
            print(f"WARNING: Genre directory not found: {genre_dir}")
            continue

        wav_files = sorted([f for f in os.listdir(genre_dir) if f.endswith(".wav")])
        print(f"Processing {genre} ({len(wav_files)} files)...")

        for wav_file in wav_files:
            file_path = os.path.join(genre_dir, wav_file)

            try:
                chunks = load_and_chunk(file_path)
            except Exception as e:
                print(f"  Skipping {wav_file}: {e}")
                skipped_files.append(wav_file)
                continue

            for chunk_num, chunk in enumerate(chunks):
                mel_spec = audio_to_mel_spectrogram(chunk)
                mel_resized = resize_spectrogram(mel_spec)

                all_spectrograms.append(mel_resized)
                all_labels.append(genre_idx)
                file_map_rows.append({
                    "chunk_index": chunk_index,
                    "genre": genre,
                    "original_file": wav_file,
                    "chunk_number": chunk_num,
                })
                chunk_index += 1

    # Convert to numpy arrays
    X = np.array(all_spectrograms, dtype=np.float32)
    X = X[..., np.newaxis]  # Add channel dimension: (N, 150, 150, 1)
    y = np.array(all_labels, dtype=np.int32)

    # Save
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    file_map_df = pd.DataFrame(file_map_rows)
    file_map_df.to_csv(os.path.join(output_dir, "file_map.csv"), index=False)

    print(f"\nPreprocessing complete:")
    print(f"  Spectrograms: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Skipped files: {len(skipped_files)}")
    if skipped_files:
        print(f"  Skipped: {skipped_files}")

    return X, y


def preprocess_single_file(file_path, sr=config.SAMPLE_RATE):
    """
    Preprocess a single audio file for inference.

    Returns array of shape (num_chunks, 150, 150, 1) ready for model.predict().
    """
    chunks = load_and_chunk(file_path, sr=sr)
    spectrograms = []

    for chunk in chunks:
        mel_spec = audio_to_mel_spectrogram(chunk, sr=sr)
        mel_resized = resize_spectrogram(mel_spec)
        spectrograms.append(mel_resized)

    X = np.array(spectrograms, dtype=np.float32)
    X = X[..., np.newaxis]
    return X


if __name__ == "__main__":
    X, y = preprocess_dataset()
