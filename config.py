import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "genres_original")
SPECTROGRAM_DIR = os.path.join(BASE_DIR, "spectrograms")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "genre_cnn.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# --- Audio ---
SAMPLE_RATE = 22050
CHUNK_DURATION = 4  # seconds
CHUNK_OVERLAP = 2   # seconds
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION   # 88200
OVERLAP_SAMPLES = SAMPLE_RATE * CHUNK_OVERLAP  # 44100

# --- Mel Spectrogram ---
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
IMG_HEIGHT = 150
IMG_WIDTH = 150

# --- Model ---
CONV_FILTERS = [32, 64, 128, 256, 512]
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
CONV_DROPOUT = 0.30
DENSE_UNITS = 1200
DENSE_DROPOUT = 0.45
NUM_CLASSES = 10

# --- Training ---
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SPLIT = 0.20
RANDOM_SEED = 42

# --- Genre labels (alphabetical, matching GTZAN folder names) ---
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
