# Migration: v1 (GTZAN) → v2 (FMA-Medium)

This document captures why we transitioned from the GTZAN-based v1 model to an FMA-Medium-based v2 model, the alternatives we evaluated, and the goals for v2.

---

## 1. v1 Model Performance (GTZAN)

The v1 model was a 5-block CNN (~11.4M parameters) trained on the GTZAN dataset (1,000 30-second clips, 10 genres). Trained for 14 epochs (early-stopped).

| Metric | Train | Validation |
|---|---|---|
| Final accuracy | **97.6%** | **74.0%** |
| Final loss | 0.07 | 1.41 |

**Diagnosis: severe overfitting.** A 24-point gap between train and val accuracy, and val loss is 20x train loss. Val accuracy is also very unstable (13% → 72% → 46% → 74% across epochs) — typical signs of:

- Too few training examples (~800 train files)
- Noisy GTZAN labels (some files are misclassified by GTZAN itself)
- Model too large for the data (~11.4M parameters)

The 74% val accuracy is OK-ish for GTZAN, but real-world performance on songs from outside the dataset is noticeably worse.

---

## 2. Why Not Just Add MTG-Jamendo to GTZAN?

MTG-Jamendo provides a `genre_tzanetakis` derived split that maps tracks to GTZAN's exact 10 genres. We investigated combining this with GTZAN:

| Genre | MTG tracks | GTZAN tracks | Combined |
|---|---:|---:|---:|
| rock | 119 | 100 | 219 |
| classical | 115 | 100 | 215 |
| pop | 61 | 100 | 161 |
| hiphop | 52 | 100 | 152 |
| metal | 24 | 100 | 124 |
| jazz | 19 | 100 | 119 |
| reggae | 7 | 100 | 107 |
| country | 6 | 100 | 106 |
| blues | 5 | 100 | 105 |
| disco | 3 | 100 | 103 |
| **Total** | **411** | **1000** | **1411** |

**Problems:**

- Only adds 411 tracks (~40% increase), not the order of magnitude we need
- Heavy imbalance toward rock/classical
- Inter-annotator agreement is only 47% (noisy labels)
- Total download is **156–508 GB** to extract those 411 tracks

**Verdict: not worth it.** It doesn't meaningfully fix the data problem.

---

## 3. Other Datasets Considered

| Option | Tracks | Genres | Pros | Cons |
|---|---|---|---|---|
| **A. FMA-Medium** ✅ | 25,000 | 16 (superset of GTZAN) | Big jump in data; covers all GTZAN genres + electronic/folk/etc.; single-label; official train/val/test splits | Larger download (~22 GB); 16-class task is harder than 10-class |
| **B. FMA-Small** | 8,000 | 8 (only 3 overlap with GTZAN) | Smaller download (~7 GB); balanced 1000/genre | Loses blues/classical/jazz/metal/etc. |
| **C. MTG-Jamendo (own 87 genre tags)** | ~55,000 | 87 (multi-label) | Massive data; full tracks | Multi-label tagging task (different from genre classification); huge download (156–508 GB) |
| **D. Stick with GTZAN + augmentation + lighter model** | 1,000 | 10 (current) | No new dataset to manage; fastest path | Caps performance at ~75–80% val acc |

**Decision: Option A (FMA-Medium).** Best ratio of effort to gain — 25× more data, single-label, and the genres are a strict superset of GTZAN.

---

## 4. FMA-Medium: Dataset Details

**Source:** [Free Music Archive Dataset (Defferrard et al., ISMIR 2017)](https://github.com/mdeff/fma)

| Property | Value |
|---|---|
| **Total tracks** | 25,000 |
| **Genres** | 16 (top-level, single-label) |
| **Audio format** | MP3, 30-second clips, 44.1 kHz |
| **Audio download size** | ~22 GB |
| **Metadata download size** | ~340 MB |
| **License** | Creative Commons (per track) |
| **Official splits** | Train / Validation / Test (provided in metadata) |

### The 16 FMA-Medium Genres

All 8 from FMA-Small:
- **Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock**

Plus 8 more in FMA-Medium:
- **Blues, Classical, Country, Easy Listening, Jazz, Old-Time/Historic, Soul-RnB, Spoken**

### GTZAN → FMA-Medium Genre Coverage

| GTZAN Genre | FMA-Medium Equivalent |
|---|---|
| blues | Blues ✅ |
| classical | Classical ✅ |
| country | Country ✅ |
| disco | Electronic / Pop (closest) |
| hiphop | Hip-Hop ✅ |
| jazz | Jazz ✅ |
| metal | Rock (subgenre, no separate label) |
| pop | Pop ✅ |
| reggae | International (subgenre) |
| rock | Rock ✅ |

### Class Distribution (Exact Counts)

FMA-Medium is **heavily unbalanced**. Per-genre track counts from the official FMA `analysis.ipynb`:

| Genre | Training | Validation | Test | **Total** |
|---|---:|---:|---:|---:|
| Rock | 5,681 | 711 | 711 | **7,103** |
| Electronic | 5,050 | 632 | 632 | **6,314** |
| Experimental | 1,801 | 225 | 225 | **2,251** |
| Hip-Hop | 1,761 | 220 | 220 | **2,201** |
| Folk | 1,215 | 152 | 152 | **1,519** |
| Instrumental | 1,045 | 131 | 174 | **1,350** |
| Pop | 945 | 122 | 119 | **1,186** |
| International | 814 | 102 | 102 | **1,018** |
| Classical | 495 | 62 | 62 | **619** |
| Old-Time / Historic | 408 | 51 | 51 | **510** |
| Jazz | 306 | 39 | 39 | **384** |
| Country | 142 | 18 | 18 | **178** |
| Soul-RnB | 94 | 18 | 42 | **154** |
| Spoken | 94 | 12 | 12 | **118** |
| Blues | 58 | 8 | 8 | **74** |
| Easy Listening | 13 | 2 | 6 | **21** |
| **TOTAL** | **19,922** | **2,505** | **2,573** | **25,000** |

**Imbalance is severe:** Rock (7,103) has ~338× more tracks than Easy Listening (21). Top 4 genres (Rock, Electronic, Experimental, Hip-Hop) account for **~71% of all tracks**.

**Implications for training:**
- We must apply **class weights** (`compute_class_weight(class_weight='balanced')`) so the model doesn't just learn to predict Rock/Electronic
- Rare classes (Easy Listening: 13 train tracks, Blues: 58, Spoken: 94) will be hard to predict reliably — even with class weights, accuracy on these will be low
- A potential **Plan B** is to drop the rarest genres (e.g., Easy Listening, Blues, Spoken) and train a 13-class model instead — would lose some coverage but improve overall accuracy

### Published Baselines

- FMA paper (2017): ~63% top-1 accuracy on FMA-Medium with CNN
- Recent works: 65–70% with stronger architectures

---

## 5. v2 Goals

1. **Train on FMA-Medium** — 25,000 tracks across 16 genres
2. **Hold out a real test set** — use FMA's official train/val/test split. The model will never see test tracks during training. Evaluation reports come from the held-out test set.
3. **Target accuracy: ~85%** — ambitious (above published baselines). Realistic stretch goals: 70–75%. If we plateau, fall back options include FMA-Small (8 genres, easier) or model architecture changes.
4. **Streamlit improvements (next phase):**
   - Accept full-length MP3 uploads (Hollywood, Bollywood, any genre)
   - Output **top-3 predicted genres** with probabilities, not just one
   - Optionally show genre timeline (which 4-second segments map to which genre)

---

## 6. What Carries Over From v1

- **Chunking pipeline** — already handles arbitrary-length audio (4s overlapping chunks). Works for full Bollywood/Hollywood tracks without modification.
- **Mel spectrogram extraction** — same approach (128 mel bands, 150×150 resized, normalized to [0,1]).
- **CNN architecture** — same 5-block design as a starting point; may simplify if overfitting persists.
- **Majority voting at inference** — same (now also returning top-N probabilities).
- **MP3 support** — librosa handles MP3 natively via ffmpeg.

---

## 7. What's Different in v2

- **Dataset:** GTZAN (1K tracks, 10 genres) → FMA-Medium (25K tracks, 16 genres)
- **Splits:** v1 stratified at file-level on full dataset → v2 uses FMA's official train/val/test splits (proper held-out test set)
- **Class weights:** added to handle FMA-Medium's class imbalance
- **Inference output:** single genre + confidence → top-3 genres with probabilities
- **No data augmentation needed** — 25× more training data should remove the need for SpecAugment-style tricks (we can add later if useful)

---

## 8. Notes

- Memory: 25sK tracks × 7 chunks each = ~175K spectrograms. I'll save as float16 to fit in Colab's 12 GB RAM.
- Chunks per track: 7 non-overlapping 4-second chunks (covers all 30s of each track).
- Splits: Use FMA's official train/val/test splits (proper held-out test set).
- Class weights: FMA-Medium is unbalanced, so I'll compute and apply class weights.
- Subset selector: Configurable so you can fall back to FMA-Small (8 genres, ~7 GB) if FMA-Medium proves too slow/big.ss
