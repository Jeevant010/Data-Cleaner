"""
Robust Audio Classification Pipeline
=====================================
1. Trims/pads audio to EXACTLY 2.0 seconds for uniform small sizes.
2. Extracts MFCC features.
3. Generates a randomized CSV with labels, durations, and features.
4. Creates a Train, Validation, and Test Split.
5. Benchmarks multiple supervised models (SVM, RF, LR, KNN, GB).
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

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ──────────────────────────  CONFIG  ──────────────────────────

BASE_DIR = Path(r"c:\Desktop\Data-Cleaner\Data")
OUTPUT_DIR = BASE_DIR / "Output"
TRIMMED_DIR = OUTPUT_DIR / "trimmed_gunshots"
CSV_PATH = OUTPUT_DIR / "dataset_features.csv"

CLASS_0_DIRS = [BASE_DIR / "audio", BASE_DIR / "sound"]
CLASS_1_DIRS = [BASE_DIR / "gun", BASE_DIR / "edge-collected-gunshot-audio"]

SR = 22050
N_MFCC = 40
TARGET_DURATION = 2.0  # Exactly 2 seconds
MAX_LEN = int(SR * TARGET_DURATION)

# Test and Val Set sizing
VAL_PER_CLASS = 75
TEST_PER_CLASS = 75

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────  HELPER FUNCTIONS  ──────────────────────

def collect_wav_files(directories: list[Path]) -> list[str]:
    files = []
    for d in directories:
        for f in d.rglob("*.wav"):
            files.append(str(f))
    return sorted(files)

def uniform_audio_length(filepath: str, output_dir: Path, is_gunshot: bool = True, top_db: int = 20) -> tuple[str, float]:
    """
    Standardize all audio files to exactly 2.0 seconds.
    If gunshot, it isolates the event first before framing.
    """
    y, sr = librosa.load(filepath, sr=SR)
    
    if is_gunshot:
        intervals = librosa.effects.split(y, top_db=top_db)
        if len(intervals) > 0:
            y = np.concatenate([y[start:end] for start, end in intervals])
            
    # Standardize length to EXACTLY MAX_LEN samples (2.0 seconds)
    if len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    else:
        y = y[:MAX_LEN]
        
    duration_ms = TARGET_DURATION * 1000.0

    stem = Path(filepath).stem
    parent_tag = Path(filepath).parent.name
    
    out_name = f"{parent_tag}__{stem}.wav"
    out_path = output_dir / out_name

    sf.write(str(out_path), y, sr)
    return str(out_path), duration_ms

def extract_mfcc(filepath: str) -> np.ndarray | None:
    try:
        y, sr = librosa.load(filepath, sr=SR)
        if len(y) < MAX_LEN:
            y = np.pad(y, (0, MAX_LEN - len(y)))
        else:
            y = y[:MAX_LEN]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])
    except Exception as e:
        print(f"  [WARN] Could not process {filepath}: {e}")
        return None

# ──────────────────────  MAIN PIPELINE  ──────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRIMMED_DIR.mkdir(parents=True, exist_ok=True)
    CLASS0_TRIMMED_DIR = OUTPUT_DIR / "trimmed_nongunshots"
    CLASS0_TRIMMED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STEP 1  ▸  Collecting audio files")
    print("=" * 65)
    class0_raw_files = collect_wav_files(CLASS_0_DIRS)
    class1_raw_files = collect_wav_files(CLASS_1_DIRS)
    print(f"  Class 0 raw files found: {len(class0_raw_files)}")
    print(f"  Class 1 raw files found: {len(class1_raw_files)}")

    print("\n" + "=" * 65)
    print(f"STEP 2  ▸  Standardizing Audio to {TARGET_DURATION} seconds")
    print("=" * 65)
    
    class0_trimmed = []
    for f in tqdm(class0_raw_files[:500], desc="  🎵 Formatting Class 0", unit="file"):
        try:
            trimmed_path, dur_ms = uniform_audio_length(f, CLASS0_TRIMMED_DIR, is_gunshot=False)
            class0_trimmed.append((trimmed_path, dur_ms))
        except Exception:
            pass

    class1_trimmed = []
    for f in tqdm(class1_raw_files[:500], desc="  🔫 Formatting Class 1", unit="file"):
        try:
            trimmed_path, dur_ms = uniform_audio_length(f, TRIMMED_DIR, is_gunshot=True)
            class1_trimmed.append((trimmed_path, dur_ms))
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("STEP 3  ▸  Extracting MFCC features")
    print("=" * 65)
    records = []
    n_features = N_MFCC * 2

    for trimmed_path, dur_ms in tqdm(class0_trimmed, desc="  🎵 Extracting Class 0", unit="file"):
        feat = extract_mfcc(trimmed_path)
        if feat is not None:
            records.append({
                "file_path": trimmed_path, "trimmed_duration_ms": dur_ms, "label": 0,
                **{f"mfcc_{i}": feat[i] for i in range(n_features)}
            })

    for trimmed_path, dur_ms in tqdm(class1_trimmed, desc="  🔫 Extracting Class 1", unit="file"):
        feat = extract_mfcc(trimmed_path)
        if feat is not None:
            records.append({
                "file_path": trimmed_path, "trimmed_duration_ms": dur_ms, "label": 1,
                **{f"mfcc_{i}": feat[i] for i in range(n_features)}
            })

    print("\n" + "=" * 65)
    print("STEP 4  ▸  Building Randomized CSV")
    print("=" * 65)
    df = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    
    n0, n1 = (df["label"] == 0).sum(), (df["label"] == 1).sum()
    print(f"  Class 0: {n0} | Class 1: {n1} | Total: {len(df)}")
    
    print("\n" + "=" * 65)
    print("STEP 5  ▸  Train / Validation / Test Splits")
    print("=" * 65)
    if n0 < (VAL_PER_CLASS+TEST_PER_CLASS) or n1 < (VAL_PER_CLASS+TEST_PER_CLASS):
        raise ValueError("Not enough samples for the desired splits.")

    df_0, df_1 = df[df["label"] == 0], df[df["label"] == 1]

    val_0 = df_0.sample(n=VAL_PER_CLASS, random_state=SEED)
    test_0 = df_0.drop(val_0.index).sample(n=TEST_PER_CLASS, random_state=SEED)
    train_0 = df_0.drop(val_0.index).drop(test_0.index)

    val_1 = df_1.sample(n=VAL_PER_CLASS, random_state=SEED)
    test_1 = df_1.drop(val_1.index).sample(n=TEST_PER_CLASS, random_state=SEED)
    train_1 = df_1.drop(val_1.index).drop(test_1.index)

    train_df = pd.concat([train_0, train_1]).sample(frac=1, random_state=SEED)
    val_df = pd.concat([val_0, val_1]).sample(frac=1, random_state=SEED)
    test_df = pd.concat([test_0, test_1]).sample(frac=1, random_state=SEED)

    feature_cols = [c for c in df.columns if c.startswith("mfcc_")]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    y_train, y_val, y_test = train_df["label"].values, val_df["label"].values, test_df["label"].values

    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    print("\n" + "=" * 65)
    print("STEP 6  ▸  Evaluating Supervised ML Benchmarks")
    print("=" * 65)

    models = {
        "Logistic Regression": LogisticRegression(random_state=SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)
    }

    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        results.append((name, acc, f1))
        
    print("\n  ---- FINAL TEST SET PERFORMANCE ----")
    print(f"  {'Model':<25} {'Accuracy':<10} {'F1-Score':<10}")
    print("  " + "-"*45)
    for name, acc, f1 in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"  {name:<25} {acc:<10.4f} {f1:<10.4f}")

    print("\n✅ Multi-Model Standardization and Training Complete!")

if __name__ == "__main__":
    main()
