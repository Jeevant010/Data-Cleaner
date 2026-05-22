# 250ms Audio Dataset Builder — Complete Manual

> **Who is this for?** Anyone with basic Python knowledge who wants to understand how we turn raw gunshot and background audio recordings into a clean, perfectly balanced dataset that machine learning models can train on without any issues.

---

## Table of Contents

1. [What Does This Notebook Do?](#1-what-does-this-notebook-do)
2. [Prerequisites](#2-prerequisites)
3. [Understanding The Data](#3-understanding-the-data)
4. [How The Pipeline Works](#4-how-the-pipeline-works)
   - [Step A: Loading Audio](#step-a-loading-audio)
   - [Step B: Extracting Gunshots (The Sniper Method)](#step-b-extracting-gunshots-the-sniper-method)
   - [Step C: Extracting Background Audio (Sliding Window)](#step-c-extracting-background-audio-sliding-window)
   - [Step D: Balancing The Dataset](#step-d-balancing-the-dataset)
5. [Running The Notebook](#5-running-the-notebook)
6. [Output Structure](#6-output-structure)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. What Does This Notebook Do?

This notebook takes your raw audio files and creates a **perfectly prepared dataset** for training machine learning models (SVM, Random Forest, CNN, TinyML, etc.) to detect gunshots.

### The Problem It Solves

Raw audio files come in all different lengths — some are 1 second, some are 10 seconds, some are 30 seconds. A machine learning model **cannot** work with files of different sizes. If you feed it different-sized files, the model will "cheat" by learning the **length** of the file instead of the **sound** inside it.

### The Solution

This notebook:
1. Cuts every single audio file into **exactly 250 millisecond** clips
2. Makes sure every gunshot clip has the "bang" perfectly centered
3. Makes sure every background clip has no gunshot sounds hiding in it
4. Creates **exactly the same number** of gunshot clips and non-gunshot clips (1:1 balance)

---

## 2. Prerequisites

### Software You Need

| Software | Why You Need It |
|----------|----------------|
| Python 3.8+ | The programming language everything runs in |
| Jupyter Notebook (or VS Code) | To open and run the `.ipynb` file |

### Python Libraries

The notebook will install these automatically when you run the first cell:

| Library | What It Does |
|---------|-------------|
| `librosa` | Loads and analyzes audio files |
| `soundfile` | Writes audio files to disk |
| `numpy` | Math operations on audio data |
| `pandas` | Creates tables and CSV reports |
| `tqdm` | Shows progress bars so you know how long to wait |
| `matplotlib` | Creates charts and plots for verification |

### How to Install Manually (if the first cell fails)

Open a terminal/command prompt and type:
```
pip install librosa soundfile pandas numpy tqdm matplotlib
```

---

## 3. Understanding The Data

### Folder Structure

Your `Data/` folder should look like this:

```
Data/
├── gun/                              ← Gunshot recordings (Class 1)
│   ├── BoltAction22_Samsung/
│   ├── Colt1911_Samsung/
│   ├── Glock9_1_Samsung/
│   ├── ... (20+ gun model folders)
│   └── WinM14_Samsung/
│
├── edge-collected-gunshot-audio/      ← More gunshot recordings (Class 1)
│   └── edge-collected-gunshot-audio/
│       ├── 38s&ws_dot38_caliber/
│       ├── glock_17_9mm_caliber/
│       ├── remington_870_12_gauge/
│       └── ruger_ar_556_dot223_caliber/
│
├── sound/                             ← Background sounds (Class 0)
│   ├── Car_Passing_Sound_Effects_1.wav
│   ├── Street_Sounds_001.wav
│   └── ... (230 files)
│
└── audio/                             ← More background sounds (Class 0)
    ├── 1-100032-A-0.wav
    ├── 1-100038-A-14.wav
    └── ... (6,000+ files)
```

### What Each Folder Contains

| Folder | Class | Count | Description |
|--------|-------|-------|-------------|
| `gun/` | 1 (Gunshot) | ~2,403 files | Samsung-recorded gunshot samples across 20+ firearm models |
| `edge-collected-gunshot-audio/` | 1 (Gunshot) | ~2,148 files | Edge-collected gunshot data from 4 caliber types |
| `sound/` | 0 (Non-Gunshot) | ~230 files | Car sounds, street ambiance, driving sounds |
| `audio/` | 0 (Non-Gunshot) | ~6,000 files | Large environmental sound dataset (dogs, rain, music, speech, etc.) |

---

## 4. How The Pipeline Works

### Step A: Loading Audio

Every audio file, regardless of its original format, is:
1. **Resampled to 22,050 Hz** (a standard rate that captures all frequencies a human can hear while keeping file sizes small)
2. **Converted to mono** (single channel — we don't need stereo for gunshot detection)
3. **Cleaned** of any corrupted values (NaN, infinity)

```
Original file (any sample rate, stereo or mono)
        ↓
Resampled to 22,050 Hz, mono
        ↓
Clean numpy array of float32 values between -1.0 and 1.0
```

### Step B: Extracting Gunshots (The Sniper Method)

This is the most important part. A gunshot is a **transient** — a sudden explosive burst that lasts only a few milliseconds, followed by a trailing echo.

#### The Old (Broken) Way: Guillotine Cutting

The old pipeline would blindly chop the audio every 250ms:
```
|---250ms---|---250ms---|---250ms---|
     ^
     If the gunshot happens HERE, it gets split across two clips.
     Neither clip has a complete gunshot → both go to "uncertain"
```

#### The New (Fixed) Way: Onset Detection

1. **Find the bang**: `librosa.onset.onset_detect()` scans the audio for sudden energy spikes
2. **Step back 50ms**: We grab 50ms of silence/quiet BEFORE the bang
3. **Grab 200ms forward**: We capture the blast and its immediate decay
4. **Result**: A perfectly centered 250ms clip where the gunshot always starts at the 50ms mark

```
|--50ms--|--------200ms---------|
  quiet    BANG + echo/decay
  ↑
  The onset is always here at 50ms

Total = 50ms + 200ms = 250ms exactly
```

#### Why This Matters For Your Model

When every gunshot is aligned the same way, the model learns to recognize the **shape of a gunshot** (sudden spike → decay), not the **position of a gunshot** (random offset). This is called **shift-invariance** and it's critical for CNN models.

### Step C: Extracting Background Audio (Sliding Window)

For non-gunshot sounds (cars, dogs, rain, speech), we use a different approach:

1. **Start at position 0** in the audio file
2. **Grab a 250ms window**
3. **Slide forward by 125ms** (50% overlap)
4. **Grab the next 250ms window**
5. **Repeat** until we reach the end of the file

```
File: |============================================|

Clip 1: |---250ms---|
Clip 2:      |---250ms---|        ← 50% overlap with Clip 1
Clip 3:           |---250ms---|   ← 50% overlap with Clip 2
...
```

#### Safety Check: Impulse Rejection

For each background clip, we check if it looks "suspiciously impulsive" — if it has a very high peak and crest factor, it might be a gunshot that accidentally ended up in the background folder. These are automatically skipped.

### Step D: Balancing The Dataset

After extracting all clips from both classes, we count them:
- If Class 1 has 10,000 clips and Class 0 has 50,000 clips → we randomly pick 10,000 from Class 0
- If Class 1 has 10,000 clips and Class 0 has 8,000 clips → we randomly pick 8,000 from Class 1

**The result is always a perfect 1:1 ratio.** This prevents the model from "cheating" by always guessing the more common class.

---

## 5. Running The Notebook

### Step-by-Step

1. **Open the notebook** in Jupyter or VS Code:
   ```
   research/The_real_Trimmer/build_clean_250ms_dataset.ipynb
   ```

2. **Run Cell 1** (Install libraries):
   ```python
   %pip install -q librosa soundfile pandas numpy tqdm matplotlib
   ```
   Wait for it to finish. You'll see "Successfully installed ..." or "Requirement already satisfied".

3. **Run Cell 2** (Import libraries):
   This loads all the tools into memory. No output expected.

4. **Run Cell 3** (Configuration):
   This auto-detects your project root. You should see:
   ```
   Project Root   : D:\Desktop\Data-Cleaner
   Data Directory : D:\Desktop\Data-Cleaner\Data
   Output         : D:\Desktop\Data-Cleaner\Data\TRIMMED_250MS_STRICT
   Clip Duration  : 250ms = 5512 samples @ 22050Hz
   ```

5. **Run Cells 4-7** (Helper functions):
   These define the extraction functions. No visible output — they just load into memory.

6. **Run Cell 8** (Build pipeline function):
   This defines the main `build_dataset()` function. No output yet.

7. **Run Cell 9** (RUN THE BUILD):
   ```python
   manifest_df = build_dataset()
   ```
   **This is the big one.** You'll see progress bars:
   ```
   Extracting Class 1 (Gunshots): 100%|████████| 4551/4551 [03:00<00:00]
   Extracting Class 0 (Non-Gunshots): 100%|████████| 6230/6230 [05:00<00:00]
   Balancing to XXXXX clips per class...
   Writing Class 1: 100%|████████|
   Writing Class 0: 100%|████████|
   ```

8. **Run Cell 10** (Verification):
   This checks every single output file to confirm it's exactly 250ms. You should see:
   ```
   ALL XXXXX files are exactly 5512 samples (250ms). Perfect!
   ```

9. **Run Cell 11** (Visual check):
   This plots a random gunshot and background clip side-by-side so you can visually verify the alignment.

### How Long Does It Take?

| Step | Time (estimate) |
|------|----------------|
| Extracting Class 1 | 2-5 minutes |
| Extracting Class 0 | 5-10 minutes |
| Writing to disk | 2-5 minutes |
| Verification | 1-3 minutes |
| **Total** | **~10-25 minutes** |

---

## 6. Output Structure

After running, your `Data/TRIMMED_250MS_STRICT/` folder will look like:

```
TRIMMED_250MS_STRICT/
├── class_1_gunshot/          ← ALL gunshot clips go here
│   ├── c1_000001.wav         ← Exactly 250ms
│   ├── c1_000002.wav         ← Exactly 250ms
│   └── ... (thousands more)
│
├── class_0_nongunshot/       ← ALL non-gunshot clips go here
│   ├── c0_000001.wav         ← Exactly 250ms
│   ├── c0_000002.wav         ← Exactly 250ms
│   └── ... (same count as class_1)
│
└── reports/
    ├── manifest.csv          ← Log of every file: label, source, path
    ├── summary.json          ← Quick stats: counts, duration, sample rate
    └── sample_comparison.png ← Visual comparison plot
```

### The manifest.csv File

This CSV is your dataset's "receipt". It tells you:

| Column | Description |
|--------|-------------|
| `label` | `1` = gunshot, `0` = not gunshot |
| `split` | `class_1_gunshot` or `class_0_nongunshot` |
| `file` | Relative path to the output WAV file |
| `source` | Which original file this clip came from |

### Using This Dataset In Your Model

To load this dataset in your training script:

```python
import pandas as pd
from pathlib import Path

dataset_root = Path('Data/TRIMMED_250MS_STRICT')
manifest = pd.read_csv(dataset_root / 'reports' / 'manifest.csv')

# All your files:
for _, row in manifest.iterrows():
    audio_path = dataset_root / row['file']
    label = row['label']  # 1 = gunshot, 0 = not gunshot
    # Load and use...
```

---

## 7. Troubleshooting

### "Could not locate project root containing Data/"

The notebook auto-searches upward from its own location for a folder containing `Data/`. Make sure:
- The `Data/` folder exists
- The notebook is somewhere inside the project (e.g., `research/The_real_Trimmer/`)

### "No Class 1 clips extracted"

This means none of the `.wav` files in `gun/` or `edge-collected-gunshot-audio/` could be loaded. Common causes:
- The folders are empty
- The files are corrupted
- The files are not `.wav` format

### "All files are not 250ms"

This should never happen with the new pipeline — `force_exact_length()` guarantees it. If you see this, please report it.

### The build takes too long

The `Data/audio/` folder has 6,000 files. Each file gets sliced into multiple 250ms clips with overlap. This is intentional — we want maximum data diversity. If you need speed, you can reduce the number of Class 0 source files.

### The model still performs badly

If the dataset is perfectly balanced and all clips are 250ms but the model still fails:
1. Check that the **sample rate** in your model matches `22050`
2. Check that your model's input size matches `5512` samples
3. Try plotting random clips from each class to verify they sound correct
4. The issue is likely in the model architecture, not the data

---

> **Remember**: A model is only as good as its data. This pipeline gives you a rock-solid foundation — every clip is the same size, every gunshot is perfectly aligned, and the classes are perfectly balanced. Now your model has no excuses!
