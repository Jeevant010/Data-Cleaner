"""
Robust Audio Classification Pipeline
=====================================
1. Trims gunshot audio (class 1) to isolate only the gunshot event.
2. Extracts MFCC features from all audio.
3. Generates a randomized CSV with labels, durations, and features.
4. Trains an SVM classifier with a balanced 150/150 test set.
5. Reports comprehensive ML evaluation metrics.
"""

import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
    r2_score,
    roc_auc_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")

# ──────────────────────────  CONFIG  ──────────────────────────

BASE_DIR = Path(r"c:\Desktop\Data-Cleaner\Data")
OUTPUT_DIR = BASE_DIR / "Output"
TRIMMED_DIR = OUTPUT_DIR / "trimmed_gunshots"
CSV_PATH = OUTPUT_DIR / "dataset_features.csv"

# Class 0 directories (non-gunshot / ambient / noise)
CLASS_0_DIRS = [BASE_DIR / "audio", BASE_DIR / "sound"]

# Class 1 directories (gunshot)
CLASS_1_DIRS = [BASE_DIR / "gun", BASE_DIR / "edge-collected-gunshot-audio"]

# MFCC feature extraction settings
SR = 22050          # Sampling rate
N_MFCC = 40         # Number of MFCC coefficients
MAX_LEN = 22050 * 4 # Pad/truncate to 4 seconds for uniformity

# Test set size per class
TEST_PER_CLASS = 150

# Random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────  HELPER FUNCTIONS  ──────────────────────


def collect_wav_files(directories: list[Path]) -> list[str]:
    """Recursively collect all .wav files from a list of directories."""
    files = []
    for d in directories:
        for f in d.rglob("*.wav"):
            files.append(str(f))
    return sorted(files)


def trim_gunshot(filepath: str, output_dir: Path,
                 top_db: int = 20) -> tuple[str, float]:
    """
    Trim leading/trailing silence from an audio file to isolate the
    gunshot event.  Returns (saved_path, duration_ms).
    """
    y, sr = librosa.load(filepath, sr=SR)

    # Use librosa to find non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        # File is entirely silent — keep as-is but flag
        trimmed = y
    else:
        # Concatenate all non-silent segments (handles multi-burst shots)
        trimmed = np.concatenate([y[start:end] for start, end in intervals])

    duration_ms = round((len(trimmed) / sr) * 1000, 2)

    # Build output path preserving a unique name
    stem = Path(filepath).stem
    # Add parent folder name to avoid collisions across gun sub-dirs
    parent_tag = Path(filepath).parent.name
    out_name = f"{parent_tag}__{stem}.wav"
    out_path = output_dir / out_name

    sf.write(str(out_path), trimmed, sr)
    return str(out_path), duration_ms


def extract_mfcc(filepath: str) -> np.ndarray | None:
    """Extract MFCC features from a .wav file, returning a fixed-length vector."""
    try:
        y, sr = librosa.load(filepath, sr=SR)

        # Pad or truncate to MAX_LEN samples
        if len(y) < MAX_LEN:
            y = np.pad(y, (0, MAX_LEN - len(y)))
        else:
            y = y[:MAX_LEN]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        # Summarise each coefficient across time: mean + std → 80 features
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])
    except Exception as e:
        print(f"  [WARN] Could not process {filepath}: {e}")
        return None


def compute_duration_ms(filepath: str) -> float:
    """Get duration of a wav file in milliseconds."""
    try:
        y, sr = librosa.load(filepath, sr=SR)
        return round((len(y) / sr) * 1000, 2)
    except Exception:
        return 0.0


# ──────────────────────  MAIN PIPELINE  ──────────────────────

def main():
    # 0. Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRIMMED_DIR.mkdir(parents=True, exist_ok=True)

    # ─── STEP 1: Collect files ───
    print("=" * 65)
    print("STEP 1  ▸  Collecting audio files")
    print("=" * 65)
    class0_files = collect_wav_files(CLASS_0_DIRS)
    class1_raw_files = collect_wav_files(CLASS_1_DIRS)
    print(f"  Class 0 (non-gunshot) files found : {len(class0_files)}")
    print(f"  Class 1 (gunshot) raw files found  : {len(class1_raw_files)}")

    # ─── STEP 2: Trim gunshot audio ───
    print("\n" + "=" * 65)
    print("STEP 2  ▸  Trimming gunshot audio (isolating gunshot events)")
    print("=" * 65)
    class1_trimmed = []  # (trimmed_path, duration_ms)
    for f in tqdm(class1_raw_files, desc="  ✂ Trimming", unit="file"):
        try:
            trimmed_path, dur_ms = trim_gunshot(f, TRIMMED_DIR)
            class1_trimmed.append((trimmed_path, dur_ms))
        except Exception as e:
            print(f"  [SKIP] {f}: {e}")

    print(f"  Trimmed gunshot clips saved: {len(class1_trimmed)}")

    # ─── STEP 3: Extract features ───
    print("\n" + "=" * 65)
    print("STEP 3  ▸  Extracting MFCC features")
    print("=" * 65)
    records = []   # list of dicts
    n_features = N_MFCC * 2  # mean + std

    # Class 0
    print("  Processing Class 0 ...")
    for f in tqdm(class0_files, desc="  🎵 Class 0", unit="file"):
        feat = extract_mfcc(f)
        if feat is not None:
            dur = compute_duration_ms(f)
            records.append({
                "file_path": f,
                "trimmed_duration_ms": dur,
                "label": 0,
                **{f"mfcc_{i}": feat[i] for i in range(n_features)},
            })

    # Class 1
    print("  Processing Class 1 (trimmed) ...")
    for trimmed_path, dur_ms in tqdm(class1_trimmed,
                                     desc="  🔫 Class 1", unit="file"):
        feat = extract_mfcc(trimmed_path)
        if feat is not None:
            records.append({
                "file_path": trimmed_path,
                "trimmed_duration_ms": dur_ms,
                "label": 1,
                **{f"mfcc_{i}": feat[i] for i in range(n_features)},
            })

    # ─── STEP 4: Build randomized CSV ───
    print("\n" + "=" * 65)
    print("STEP 4  ▸  Building randomized CSV")
    print("=" * 65)
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)

    n0 = (df["label"] == 0).sum()
    n1 = (df["label"] == 1).sum()
    print(f"  Total samples : {len(df)}")
    print(f"  Class 0       : {n0}")
    print(f"  Class 1       : {n1}")
    print(f"  CSV saved     : {CSV_PATH}")

    # ─── STEP 5: Balanced train/test split ───
    print("\n" + "=" * 65)
    print("STEP 5  ▸  Creating balanced 150/150 train-test split")
    print("=" * 65)

    if n0 < TEST_PER_CLASS or n1 < TEST_PER_CLASS:
        raise ValueError(
            f"Not enough samples! Need at least {TEST_PER_CLASS} per class. "
            f"Got class0={n0}, class1={n1}."
        )

    df_0 = df[df["label"] == 0].copy()
    df_1 = df[df["label"] == 1].copy()

    # Sample TEST_PER_CLASS from each class for the test set
    test_0 = df_0.sample(n=TEST_PER_CLASS, random_state=SEED)
    test_1 = df_1.sample(n=TEST_PER_CLASS, random_state=SEED)
    test_df = pd.concat([test_0, test_1]).sample(frac=1, random_state=SEED)

    train_df = df.drop(test_df.index).sample(frac=1, random_state=SEED)

    feature_cols = [c for c in df.columns if c.startswith("mfcc_")]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"  Train set size : {len(train_df)}  "
          f"(Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()})")
    print(f"  Test  set size : {len(test_df)}  "
          f"(Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()})")

    # ─── STEP 6: Scale features ───
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ─── STEP 7: Train SVM ───
    print("\n" + "=" * 65)
    print("STEP 6  ▸  Training SVM model")
    print("=" * 65)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              random_state=SEED)
    svm.fit(X_train, y_train)
    print("  ✅ SVM training complete.")

    # ─── STEP 8: Evaluation ───
    print("\n" + "=" * 65)
    print("STEP 7  ▸  Evaluation Metrics")
    print("=" * 65)

    y_pred = svm.predict(X_test)
    y_proba = svm.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    r2 = r2_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n  {'Metric':<30} {'Score':>10}")
    print("  " + "-" * 42)
    print(f"  {'Accuracy':<30} {acc:>10.4f}")
    print(f"  {'Precision':<30} {prec:>10.4f}")
    print(f"  {'Recall':<30} {rec:>10.4f}")
    print(f"  {'F1 Score':<30} {f1:>10.4f}")
    print(f"  {'Log Loss':<30} {logloss:>10.4f}")
    print(f"  {'R² Score':<30} {r2:>10.4f}")
    print(f"  {'ROC-AUC':<30} {roc_auc:>10.4f}")
    print(f"  {'Matthews Corr. Coeff.':<30} {mcc:>10.4f}")

    print("\n  ── Classification Report ──\n")
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Gunshot (0)",
                                              "Gunshot (1)"]))

    print("  ── Confusion Matrix ──\n")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                    Predicted 0   Predicted 1")
    print(f"  Actual 0 (Noise) :    {cm[0][0]:<12d} {cm[0][1]}")
    print(f"  Actual 1 (Gun)   :    {cm[1][0]:<12d} {cm[1][1]}")

    print("\n" + "=" * 65)
    print("✅  Pipeline complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
