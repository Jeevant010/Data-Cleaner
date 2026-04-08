import os
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import random

BASE_DIR = Path(r"c:\Desktop\Data-Cleaner\Data")
OUTPUT_DIR = BASE_DIR / "Output"
EDA_DIR = OUTPUT_DIR / "EDA"
EDA_DIR.mkdir(parents=True, exist_ok=True)

CLASS_0_DIRS = [BASE_DIR / "audio", BASE_DIR / "sound"]
CLASS_1_DIRS = [BASE_DIR / "gun", BASE_DIR / "edge-collected-gunshot-audio"]

def get_files(directories):
    files = []
    for d in directories:
        for f in d.rglob("*.wav"):
            files.append(str(f))
    return sorted(files)

c0_files = get_files(CLASS_0_DIRS)
c1_files = get_files(CLASS_1_DIRS)
print(f"Found {len(c0_files)} Class 0 and {len(c1_files)} Class 1 files.")

def get_durations(files, sample_size=200):
    durations = []
    samples = random.sample(files, min(len(files), sample_size))
    for f in samples:
        try:
            y, sr = librosa.load(f, sr=None)
            durations.append(len(y) / sr)
        except:
            pass
    return durations

print("Sampling durations...")
d0 = get_durations(c0_files)
d1 = get_durations(c1_files)

plt.figure(figsize=(10, 5))
plt.hist(d0, bins=30, alpha=0.5, label='Class 0 (Non-Gunshot)', density=True)
plt.hist(d1, bins=30, alpha=0.5, label='Class 1 (Gunshot)', density=True)
plt.title("Distribution of Audio Durations (Before Trimming)")
plt.xlabel("Duration (seconds)")
plt.ylabel("Density")
plt.legend()
plt.savefig(EDA_DIR / "duration_distribution.png")
plt.close()

# Plot typical waveform and spectrogram
def plot_audio_features(file_path, title, out_name):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"{title} - Waveform")
        
        plt.subplot(2, 1, 2)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{title} - Mel-Spectrogram")
        
        plt.tight_layout()
        plt.savefig(EDA_DIR / out_name)
        plt.close()
    except Exception as e:
        print(f"Error plotting {file_path}: {e}")

if c0_files:
    plot_audio_features(c0_files[0], "Sample Class 0 (Non-Gunshot)", "sample_c0.png")
if c1_files:
    plot_audio_features(c1_files[0], "Sample Class 1 (Gunshot)", "sample_c1.png")

print("EDA plots exported to Data/Output/EDA.")
