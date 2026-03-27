# Music Genre Classification Using Deep Learning and Mel Spectrograms

A CNN-based music genre classification system that processes raw audio files and classifies them into 10 genres using Mel spectrogram features.

## Genres

blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

## Project Structure

```
config.py            # All hyperparameters and path constants
setup_dataset.py     # Download and validate GTZAN dataset
preprocess.py        # Audio chunking + Mel spectrogram extraction
model.py             # CNN architecture (5 conv blocks, ~11.4M params)
train.py             # Training with file-level stratified split
evaluate.py          # Chunk-level and file-level evaluation
inference.py         # Single-file prediction with majority voting
app.py               # Streamlit web application
notebooks/
  train_colab.ipynb  # Google Colab GPU training notebook
```

## Setup

### Prerequisites

- Python 3.12 (recommended) or 3.10+
- pip

### 1. Clone and create virtual environment

```bash
git clone https://github.com/mahalakshmi-a-iyer/music_genre_prediction.git
cd music_genre_prediction
python3.12 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Download the GTZAN dataset

**Option A** - Using kagglehub (recommended):
```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.login()"
python setup_dataset.py
```

**Option B** - Using environment variables:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
python setup_dataset.py
```

**Option C** - Manual download:
1. Download from https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
2. Extract so that `data/genres_original/blues/blues.00000.wav` exists (10 genre folders)
3. Run `python setup_dataset.py` to validate

### 3. Preprocess audio into spectrograms (~5-10 min)

```bash
python preprocess.py
```

This generates `spectrograms/X.npy`, `spectrograms/y.npy`, and `spectrograms/file_map.csv`.

### 4. Train the model

**Option A** - Google Colab (recommended, ~10-15 min with GPU):
1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Set runtime to GPU: `Runtime > Change runtime type > T4 GPU`
3. Run all cells
4. Download `models/genre_cnn.keras` and `models/history.json` to your local `models/` folder

**Option B** - Train locally (slow on CPU, ~60+ min):
```bash
python train.py
```

### 5. Evaluate the model

```bash
python evaluate.py
```

This prints chunk-level and file-level (majority voting) accuracy and saves plots to `plots/`.

Note: If `X_test.npy` doesn't exist yet, the script will automatically generate the test split from the full dataset.

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Upload a `.wav` file to get a genre prediction.

### 7. Test inference on a specific file

```bash
python inference.py data/genres_original/jazz/jazz.00000.wav
```

## How It Works

1. **Audio Chunking**: Each 30s audio track is split into 4-second chunks with 2-second overlap (~13 chunks per track)
2. **Feature Extraction**: Mel spectrograms (128 bands) are computed and resized to 150x150 pixels
3. **CNN Classification**: A 5-block CNN with BatchNormalization predicts genre per chunk
4. **Majority Voting**: Final genre is determined by majority vote across all chunks

## Known Issues

- `jazz.00054.wav` in the GTZAN dataset is corrupted and is automatically skipped during preprocessing
- On macOS with Python 3.9, TensorFlow may crash with mutex errors. Use Python 3.12+
- If using Colab for training, make sure to download both `genre_cnn.keras` and `history.json` from the `models/` folder

## Tech Stack

Python, TensorFlow/Keras, Librosa, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit
