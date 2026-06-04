# Non-Gunshot Trimmer — Complete Manual

> **Who is this for?** Anyone who wants to extract clean background audio clips that are guaranteed to contain NO gunshot-like sounds, for use as the negative class in a gunshot detection model.

---

## Table of Contents

1. [What Does This Notebook Do?](#1-what-does-this-notebook-do)
2. [Prerequisites](#2-prerequisites)
3. [How to Change Paths](#3-how-to-change-paths)
4. [How The Pipeline Works](#4-how-the-pipeline-works)
5. [Safety Checks Explained](#5-safety-checks-explained)
6. [Running The Notebook](#6-running-the-notebook)
7. [Output Structure](#7-output-structure)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What Does This Notebook Do?

This notebook scans all non-gunshot audio sources (car sounds, street ambience, environmental sounds) and extracts clean 250ms clips using a sliding window approach. Every clip is checked for "suspicious" gunshot-like characteristics and rejected if it looks impulsive.

### Why This Matters

If a clip with a car backfire or door slam ends up in the "non-gunshot" class, your model learns that **gunshot-like transients are NOT gunshots**. This confuses the model and destroys accuracy. This trimmer prevents that.

---

## 2. Prerequisites

Same as Gunshot Trimmer. The notebook auto-installs:
```
pip install librosa soundfile pandas numpy tqdm matplotlib
```

---

## 3. How to Change Paths

### Where To Change (Cell 3 — Configuration)

```python
# --- Path Auto-Detection ---
PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / 'Data').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

# !! TO HARDCODE: Uncomment below !!
# PROJECT_ROOT = Path(r'd:\Desktop\Data-Cleaner')

DATA_DIR = PROJECT_ROOT / 'Data'
OUTPUT_DIR = DATA_DIR / 'TRIMMED_NONGUNSHOTS'
```

### Changing Source Directories

```python
CLASS0_DIRS = [DATA_DIR / 'sound']                    # Default: just Data/sound/
INCLUDE_AUDIO_FOLDER = True                           # Set True to also include Data/audio/
```

### Adding Custom Source Folders

```python
CLASS0_DIRS = [
    DATA_DIR / 'sound',
    Path(r'E:\MyBackgroundSounds\city_noise'),        # Add extra sources
]
```

---

## 4. How The Pipeline Works

### Sliding Window Extraction

Unlike gunshots (which need onset detection), background audio is continuous. We use a sliding window:

```
Source file: ════════════════════════════════════════════

Window 1:  |─── 250ms ───|
Window 2:       |─── 250ms ───|        ← 125ms hop (50% overlap)
Window 3:            |─── 250ms ───|
Window 4:                 |─── 250ms ───|
...
```

The 50% overlap ensures we don't miss transitions and gives us maximum data diversity.

### Safety Filter

Each window passes through three filters:
1. **Silence filter**: Skip if `peak < 0.003` (pure silence)
2. **Impulse filter**: Skip if `crest_factor > 8.5 AND peak > 0.1` (looks like a gunshot)
3. **Spectral guard**: Skip if high centroid + high attack ratio (suspicious transient)

### Diversity Sampling

To prevent one long recording from dominating the dataset:
- Max **40 clips per source file**
- Windows are randomly shuffled before selection
- Ensures variety across different sound types

---

## 5. Safety Checks Explained

| Check | Threshold | What It Catches |
|-------|-----------|-----------------|
| **Silence** | `peak < 0.003` | Digital silence, dead air |
| **Impulse rejection** | `crest > 8.5 AND peak > 0.1` | Car backfires, door slams, possible gunshots |
| **High centroid + attack** | `centroid > 5000 AND attack_ratio > 10` | Sharp transients that could confuse the model |

Clips failing the impulse check go to `suspicious/` (not deleted), so you can manually review them.

---

## 6. Running The Notebook

| Cell | What It Does | Time |
|------|-------------|------|
| 1 | Install dependencies | ~10s |
| 2 | Import libraries | ~2s |
| 3 | Configuration (check paths!) | instant |
| 4 | Define helper functions | instant |
| 5 | Define background extraction | instant |
| 6 | Define safety checks | instant |
| 7 | **RUN THE BUILD** | 5–20 min |
| 8 | Verification | 1–5 min |
| 9 | Visual audit | ~10s |
| 10 | Balance check vs gunshot trimmer | ~5s |

---

## 7. Output Structure

```
Data/TRIMMED_NONGUNSHOTS/
├── verified/            ← All clean non-gunshot clips
│   ├── c0_000001_sound_Street_Sounds_w000.wav
│   └── ... (thousands more)
├── suspicious/          ← Clips that tripped the impulse filter
│   ├── sus_000001_impulse.wav
│   └── ...
└── reports/
    ├── manifest.csv
    └── summary.json
```

---

## 8. Troubleshooting

### Too many clips rejected as suspicious
Your background recordings might have loud transients. Lower the impulse threshold:
```python
CREST_IMPULSE = 10.0    # was 8.5 — raise to be more permissive
PEAK_IMPULSE = 0.15     # was 0.1
```

### Not enough clips extracted
- Enable `INCLUDE_AUDIO_FOLDER = True` to include the ~6,000 files in `Data/audio/`
- Raise `MAX_CLIPS_PER_FILE` from 40 to a higher number

### "All my clips sound the same"
If most clips come from one long recording, reduce `MAX_CLIPS_PER_FILE` to spread across more source files.

---

> **Remember**: A non-gunshot dataset with even one hidden gunshot will teach your model to IGNORE that sound. This trimmer ensures your negative class is completely clean.
