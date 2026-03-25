"""
Streamlit web application for music genre classification.

Usage:
    streamlit run app.py
"""

import os
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

import config
from inference import predict_genre


@st.cache_resource
def get_model():
    """Load the trained model once and cache across sessions."""
    model_path = config.MODEL_PATH

    if not os.path.exists(model_path):
        # Try downloading from Hugging Face Hub
        try:
            from huggingface_hub import hf_hub_download
            st.info("Downloading model... This only happens once.")
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            hf_hub_download(
                repo_id=os.environ.get("HF_MODEL_REPO", "your-team/music-genre-cnn"),
                filename="genre_cnn.keras",
                local_dir=config.MODEL_DIR,
            )
        except Exception as e:
            st.error(
                f"Could not load model. Place `genre_cnn.keras` in the "
                f"`models/` directory or set the HF_MODEL_REPO env var.\n\n"
                f"Error: {e}"
            )
            st.stop()

    from tensorflow import keras
    return keras.models.load_model(model_path)


def plot_mel_spectrogram(file_path):
    """Generate a Mel spectrogram visualization for the uploaded file."""
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, res_type="kaiser_fast")

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS,
        hop_length=config.HOP_LENGTH, n_fft=config.N_FFT,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=config.HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax,
    )
    ax.set_title("Mel Spectrogram")
    plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Music Genre Classifier",
        page_icon="🎵",
        layout="centered",
    )

    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown(
            "This app classifies music into **10 genres** using a CNN "
            "trained on Mel spectrograms from the GTZAN dataset."
        )
        st.markdown("**Genres:** " + ", ".join(config.GENRES))
        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown(
            "1. Audio is split into 4-second chunks\n"
            "2. Mel spectrograms are extracted\n"
            "3. CNN predicts genre per chunk\n"
            "4. Majority voting gives final result"
        )
        st.markdown("---")
        st.markdown("Deep Learning Project")

    # Main area
    st.title("Music Genre Classifier")
    st.markdown("Upload a `.wav` file to predict its genre.")

    uploaded_file = st.file_uploader(
        "Choose a WAV file", type=["wav"], label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Save to temp file (librosa needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Audio player
            st.audio(uploaded_file, format="audio/wav")

            # Load model and predict
            model = get_model()

            with st.spinner("Analyzing audio..."):
                result = predict_genre(tmp_path, model=model)

            # Results
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Genre", result["predicted_genre"].upper())
            with col2:
                st.metric("Confidence", f"{result['confidence']:.0%}")

            # Genre probability bar chart
            st.markdown("### Genre Probabilities")
            prob_df = pd.DataFrame({
                "Genre": list(result["genre_probabilities"].keys()),
                "Probability": list(result["genre_probabilities"].values()),
            }).set_index("Genre")
            st.bar_chart(prob_df)

            # Mel spectrogram visualization
            st.markdown("### Mel Spectrogram")
            fig = plot_mel_spectrogram(tmp_path)
            st.pyplot(fig)
            plt.close(fig)

            # Chunk details (expandable)
            with st.expander("Chunk-level predictions"):
                chunk_df = pd.DataFrame({
                    "Chunk": range(1, len(result["chunk_predictions"]) + 1),
                    "Predicted Genre": result["chunk_predictions"],
                })
                st.dataframe(chunk_df, use_container_width=True, hide_index=True)

        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
