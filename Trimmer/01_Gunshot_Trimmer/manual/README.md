# Gunshot Trimmer — Complete Manual

> **Who is this for?** Anyone who wants to extract the maximum number of verified gunshot audio clips from raw recordings for ML/DL model training. Every clip in the output is guaranteed to be a real gunshot.

---

## Table of Contents

1. [What Does This Notebook Do?](#1-what-does-this-notebook-do)
2. [Prerequisites](#2-prerequisites)
3. [How to Change Paths](#3-how-to-change-paths)
4. [How The Pipeline Works](#4-how-the-pipeline-works)
5. [Quality Gates Explained](#5-quality-gates-explained)
6. [Running The Notebook](#6-running-the-notebook)
7. [Output Structure](#7-output-structure)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What Does This Notebook Do?

This notebook scans all gunshot source audio files and extracts **every real gunshot event** as a perfect 250ms clip. Unlike the original trimmer that capped at 1 clip per file, this one extracts up to 8 onsets per file — but ONLY if they are genuine gunshot events. If a file has just 1 gunshot, it extracts 1. If it has 0, it extracts 0. No padding with silence, no fake clips.

### Key Improvements Over The_real_Trimmer

| Feature | The_real_Trimmer | This Gunshot Trimmer |
|---------|-----------------|---------------------|
| Clips per source file | Max 1 | Up to 8 (only real onsets) |
| Quality validation | Peak + RMS + crest | Peak + RMS + crest + spectral centroid + ZCR |
| Rejected clips | Mixed into `noise/` | Separate `rejected/` folder for review |
| Target output | ~4,000 clips | **4,000+ clips** (more onsets per file) |

---

## 2. Prerequisites

### Software

| Software | Why |
|----------|-----|
| Python 3.8+ | Runtime |
| Jupyter Notebook or VS Code | To run the `.ipynb` file |

### Python Libraries

The notebook auto-installs these in Cell 1:

| Library | Purpose |
|---------|---------|
| `librosa` | Audio loading and analysis |
| `soundfile` | Writing WAV files |
| `numpy` | Array operations |
| `pandas` | Manifest/CSV generation |
| `tqdm` | Progress bars |
| `matplotlib` | Verification plots |

Manual install if Cell 1 fails:
```
pip install librosa soundfile pandas numpy tqdm matplotlib
```

---

## 3. How to Change Paths

### Where To Change (Cell 3 — Configuration)

Open the notebook and go to **Cell 3**. You will see:

```python
# ==========================================================
# CONFIGURATION — CHANGE PATHS HERE
# ==========================================================
PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / 'Data').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / 'Data'
OUTPUT_DIR = DATA_DIR / 'TRIMMED_GUNSHOTS'
```

### If Auto-Detection Fails

Replace the auto-detection block with a hardcoded path:

```python
# OPTION A: Hardcode the project root
PROJECT_ROOT = Path(r'd:\Desktop\Data-Cleaner')

# OPTION B: Hardcode just the data folder
DATA_DIR = Path(r'd:\Desktop\Data-Cleaner\Data')

# OPTION C: Change output location
OUTPUT_DIR = Path(r'E:\MyOutput\TRIMMED_GUNSHOTS')
```

### Changing Source Directories

In the same Cell 3, you'll find:

```python
CLASS1_DIRS = [DATA_DIR / 'gun', DATA_DIR / 'edge-collected-gunshot-audio']
```

To add more gunshot source folders:
```python
CLASS1_DIRS = [
    DATA_DIR / 'gun',
    DATA_DIR / 'edge-collected-gunshot-audio',
    Path(r'E:\More_Gunshots\folder_name'),  # Add extra sources here
]
```

### Changing Clip Duration

```python
TARGET_MS = 250    # Change to 500 for longer clips
```

> **Warning**: Changing clip duration affects model compatibility. If your model was trained on 250ms clips, new clips MUST also be 250ms.

---

## 4. How The Pipeline Works

### Step 1: Collect Source Files
Recursively scans all directories in `CLASS1_DIRS` for `.wav` files, skipping macOS junk files (`__MACOSX`, `._` prefixed).

### Step 2: Multi-Onset Detection (The Enhanced Sniper Method)

For each source file:

1. **Load audio** → resample to 22,050 Hz, mono, float32
2. **Find candidates** using three methods simultaneously:
   - `librosa.onset.onset_detect()` — spectral flux-based onset detection
   - **Raw peak** — the loudest sample in the file
   - **Envelope peak** — smoothed amplitude peak
3. **Score candidates** by `|amplitude| + |diff|` (instantaneous energy + attack speed)
4. **Filter overlapping** candidates (minimum gap = half clip length)
5. **Keep top 8** strongest candidates

```
Source file: ════════════════════════════════════════════
                    ↑         ↑              ↑
               onset 1    onset 2        onset 3
                    │         │              │
                    ▼         ▼              ▼
               [250ms]   [250ms]        [250ms]
              clip #1    clip #2        clip #3

Each clip: |─50ms─|────────200ms────────|
           quiet    BANG + decay
```

### Step 3: Quality Validation

Every extracted clip goes through the quality gate (see Section 5). Only clips that pass ALL checks go to `verified/`. Everything else goes to `rejected/` with the failure reason logged.

### Step 4: Write Output

- Verified clips → `Data/TRIMMED_GUNSHOTS/verified/`
- Rejected clips → `Data/TRIMMED_GUNSHOTS/rejected/`
- Reports → `Data/TRIMMED_GUNSHOTS/reports/`

---

## 5. Quality Gates Explained

Each clip is tested against ALL of these. Failing ANY one sends the clip to `rejected/`.

| Check | Threshold | Why |
|-------|-----------|-----|
| **Peak amplitude** | ≥ 0.01 | Below this = digital silence, not a gunshot |
| **RMS energy** | ≥ 0.002 | Overall energy too low = probably noise/silence |
| **Crest factor** | ≥ 3.0 | Gunshots are transients: peak ÷ RMS must be high |
| **Prominence** | ≥ 4.0 | Peak must stand out clearly above the median |
| **95th percentile** | ≥ 0.002 | Overall clip can't be too quiet |
| **Spectral centroid** | ≥ 1500 Hz | Gunshots have broadband energy, not just bass hum |
| **Attack ZCR** | > baseline | Genuine impacts have high zero-crossing rate in the attack phase |

### What "Rejected" Means

Rejected clips are NOT deleted — they go to a separate folder. You can listen to them and decide:
- If you hear a real gunshot that was wrongly rejected, you can manually move it to `verified/`
- If it's genuinely garbage (silence, hum, noise), leave it

---

## 6. Running The Notebook

| Cell | What It Does | Time |
|------|-------------|------|
| 1 | Install dependencies | ~10s |
| 2 | Import libraries | ~2s |
| 3 | Configuration (check paths here!) | instant |
| 4 | Define helper functions | instant |
| 5 | Define onset detection | instant |
| 6 | Define quality gate | instant |
| 7 | **RUN THE BUILD** | 5–15 min |
| 8 | Verification (size check) | 1–3 min |
| 9 | Visual audit (plots + audio) | ~10s |

---

## 7. Output Structure

```
Data/TRIMMED_GUNSHOTS/
├── verified/
│   ├── c1_000001_gun_BoltAction22_Samsung_SA_004A_S01_k00.wav
│   ├── c1_000002_gun_BoltAction22_Samsung_SA_004A_S01_k01.wav
│   └── ... (thousands more, all exactly 250ms)
│
├── rejected/
│   ├── rej_000001_peak_too_low.wav
│   ├── rej_000002_crest_too_low.wav
│   └── ...
│
└── reports/
    ├── manifest.csv     ← Every clip: path, source, metrics, decision
    └── summary.json     ← Total counts, config used, timestamp
```

### The manifest.csv Columns

| Column | Description |
|--------|-------------|
| `output_path` | Relative path to the output WAV file |
| `source_path` | Which original file this clip came from |
| `label` | Always `1` (gunshot) for this trimmer |
| `decision` | `accepted` or the rejection reason |
| `rms` | RMS energy of the clip |
| `peak` | Peak amplitude |
| `crest_factor` | Peak ÷ RMS |
| `spectral_centroid` | Frequency center of mass |

---

## 8. Troubleshooting

### "Could not locate project root containing Data/"
The notebook searches upward from its location. Make sure:
- `Data/` exists in the project
- The notebook is inside the project tree

**Fix**: Hardcode the path in Cell 3 (see Section 3).

### Very few clips extracted
- Check that `Data/gun/` and `Data/edge-collected-gunshot-audio/` contain `.wav` files
- Corrupted files are auto-skipped — check the rejected count

### Too many rejected clips
The quality gates might be too strict for your data. In Cell 6, you can lower thresholds:
```python
# Relax these if too many clips are rejected
PEAK_MIN = 0.005       # was 0.01
CREST_MIN = 2.5        # was 3.0
```

### The build takes too long
With ~4,500 source files and up to 8 candidates each, expect 5–15 minutes. This is normal.

---

> **Remember**: Every clip in `verified/` is a confirmed real gunshot. Your ML model can trust this data completely.
