"""
Download and validate the GTZAN Genre Collection dataset.

Usage:
    python setup_dataset.py

Supports:
    1. Automatic download via kagglehub (recommended, uses API tokens)
    2. Automatic download via Kaggle CLI (legacy kaggle.json)
    3. Manual download with printed instructions
"""

import os
import sys
import shutil

import config


def download_via_kagglehub(dest_dir="data"):
    """
    Download GTZAN using kagglehub (recommended for Kaggle CLI >= 1.8.0).

    Uses API tokens — authenticate with: kaggle auth login
    """
    try:
        import kagglehub
    except ImportError:
        return None

    print("Downloading GTZAN dataset via kagglehub...")
    try:
        path = kagglehub.dataset_download(
            "andradaolteanu/gtzan-dataset-music-genre-classification"
        )
        print(f"Downloaded to cache: {path}")
        return path
    except Exception as e:
        print(f"kagglehub download failed: {e}")
        return None


def download_via_kaggle_cli(dest_dir="data"):
    """Download GTZAN using legacy Kaggle CLI (kaggle.json)."""
    import subprocess
    import zipfile

    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return None
    except FileNotFoundError:
        return None

    zip_path = os.path.join(dest_dir, "gtzan-dataset-music-genre-classification.zip")

    print("Downloading GTZAN dataset via Kaggle CLI...")
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "andradaolteanu/gtzan-dataset-music-genre-classification",
                "-p", dest_dir,
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Kaggle CLI download failed.")
        return None

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    return dest_dir


def organize_dataset(source_dir, dest_dir="data"):
    """
    Move genre folders to data/genres_original/ regardless of
    where kagglehub or CLI extracted them.
    """
    target_path = os.path.join(dest_dir, "genres_original")

    if os.path.isdir(target_path) and len(os.listdir(target_path)) >= 10:
        return target_path

    # Search common extraction layouts for the genres_original folder
    candidates = [
        os.path.join(source_dir, "Data", "genres_original"),
        os.path.join(source_dir, "genres_original"),
        os.path.join(source_dir, "data", "genres_original"),
    ]

    # Also check if source_dir itself contains genre folders directly
    if os.path.isdir(os.path.join(source_dir, "blues")):
        candidates.insert(0, source_dir)

    found = None
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "blues")):
            found = candidate
            break

    if found is None:
        print(f"WARNING: Could not find genre folders in {source_dir}")
        print("Please manually place genre folders in data/genres_original/")
        return None

    if os.path.abspath(found) != os.path.abspath(target_path):
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(found, target_path)

    # Clean up extra directories
    data_subdir = os.path.join(dest_dir, "Data")
    if os.path.isdir(data_subdir):
        shutil.rmtree(data_subdir)

    return target_path


def download_gtzan(dest_dir="data"):
    """Download and organize the GTZAN dataset."""
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.isdir(config.DATA_DIR):
        genre_count = len([
            d for d in os.listdir(config.DATA_DIR)
            if os.path.isdir(os.path.join(config.DATA_DIR, d))
            and not d.startswith(".")
        ])
        if genre_count == 10:
            print("Dataset already exists. Skipping download.")
            return config.DATA_DIR

    # Try kagglehub first (newer API token method)
    source = download_via_kagglehub(dest_dir)

    # Fall back to legacy CLI
    if source is None:
        source = download_via_kaggle_cli(dest_dir)

    if source is None:
        print_manual_instructions()
        sys.exit(1)

    result = organize_dataset(source, dest_dir)
    if result:
        print(f"Dataset ready at: {result}")
    return result


def print_manual_instructions():
    """Print manual download instructions."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("Option 1 - kagglehub (recommended):")
    print("  1. pip install kagglehub")
    print("  2. Run: kaggle auth login")
    print("     (follow the prompts to authenticate)")
    print("  3. Run this script again")
    print()
    print("Option 2 - Kaggle CLI (legacy):")
    print("  1. pip install kaggle")
    print("  2. Go to kaggle.com -> Account -> Create API Token")
    print("  3. Place kaggle.json in ~/.kaggle/")
    print("  4. Run this script again")
    print()
    print("Option 3 - Manual download:")
    print("  1. Go to: https://www.kaggle.com/datasets/"
          "andradaolteanu/gtzan-dataset-music-genre-classification")
    print("  2. Download the dataset")
    print("  3. Extract so that the folder structure is:")
    print("     data/genres_original/blues/blues.00000.wav")
    print("     data/genres_original/classical/classical.00000.wav")
    print("     ... (10 genre folders, ~100 .wav files each)")
    print("=" * 60)


def validate_dataset(data_dir=None):
    """Validate that the dataset is correctly structured."""
    if data_dir is None:
        data_dir = config.DATA_DIR

    print(f"\nValidating dataset at: {data_dir}")
    print("-" * 45)

    if not os.path.isdir(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        return False

    genres_found = sorted(os.listdir(data_dir))
    genres_found = [g for g in genres_found if not g.startswith(".")]

    valid = True
    total_files = 0

    for genre in config.GENRES:
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_dir):
            print(f"  MISSING: {genre}/")
            valid = False
            continue

        wav_files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
        count = len(wav_files)
        total_files += count
        status = "OK" if count >= 90 else "LOW"
        print(f"  {genre:12s} : {count:3d} files  [{status}]")

    print("-" * 45)
    print(f"  Total: {total_files} files across {len(genres_found)} genres")

    if valid and total_files >= 900:
        print("  Dataset validation: PASSED")
    else:
        print("  Dataset validation: FAILED")
        valid = False

    return valid


if __name__ == "__main__":
    download_gtzan()
    validate_dataset()
