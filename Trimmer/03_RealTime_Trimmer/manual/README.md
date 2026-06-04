# Real-Time / Test Data Trimmer — Complete Manual

> **Who is this for?** Anyone who wants to process test audio data through a clean extraction pipeline, or capture live audio from a microphone for real-time gunshot event detection.

---

## Table of Contents

1. [What Does This Notebook Do?](#1-what-does-this-notebook-do)
2. [Prerequisites](#2-prerequisites)
3. [The Three Modes](#3-the-three-modes)
4. [How to Change Paths](#4-how-to-change-paths)
5. [Microphone Setup (Windows)](#5-microphone-setup-windows)
6. [Running The Notebook](#6-running-the-notebook)
7. [Output Structure](#7-output-structure)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What Does This Notebook Do?

This is a **multi-mode trimmer** with three operating modes. You switch between them by setting one variable:

```python
MODE = "batch"       # Process a folder of files
MODE = "single"      # Process one file with detailed analysis
MODE = "microphone"  # Capture live audio from your PC's mic
```

Each mode extracts 250ms clips and classifies them using audio features (energy, crest factor, spectral centroid). No ML model is needed — this uses heuristic-based detection.

---

## 2. Prerequisites

### Software

| Software | Why |
|----------|-----|
| Python 3.8+ | Runtime |
| Jupyter Notebook or VS Code | To run the notebook |

### Python Libraries

The notebook auto-installs these. **Microphone mode** requires `sounddevice`:

```
pip install librosa soundfile pandas numpy tqdm matplotlib sounddevice
```

> **Note**: `sounddevice` requires PortAudio. On Windows, it's included automatically. On Linux, you may need: `sudo apt-get install libportaudio2`

---

## 3. The Three Modes

### Mode 1: `batch`

| Aspect | Detail |
|--------|--------|
| **Input** | A folder of `.wav` files |
| **Processing** | Onset detection + sliding window |
| **Classification** | Heuristic: high crest + high energy = gunshot |
| **Output** | Clips sorted into `gunshot/` and `nongunshot/` subfolders |
| **Best for** | Processing a batch of unseen test recordings |

### Mode 2: `single`

| Aspect | Detail |
|--------|--------|
| **Input** | One `.wav` file path |
| **Processing** | Full onset detection + every window analyzed |
| **Output** | Detailed report: waveform, spectrogram, detected events, extracted clips |
| **Best for** | Deep-diving into a single recording to understand what's in it |

### Mode 3: `microphone`

| Aspect | Detail |
|--------|--------|
| **Input** | Live audio from your PC microphone |
| **Processing** | Rolling buffer with energy-based trigger |
| **Output** | Captured event clips saved automatically |
| **Best for** | Collecting live test data, real-time demo |

---

## 4. How to Change Paths

### Configuration Cell (Cell 3)

```python
# ==========================================================
# MODE SELECTION — Change this to switch modes
# ==========================================================
MODE = "batch"           # Options: "batch", "single", "microphone"

# ==========================================================
# PATHS — Change these based on your mode
# ==========================================================
PROJECT_ROOT = Path(r'd:\Desktop\Data-Cleaner')

# For BATCH mode: folder containing .wav files to process
BATCH_INPUT_DIR = Path(r'd:\Desktop\Data-Cleaner\Data\gun')

# For SINGLE mode: one .wav file to analyze
SINGLE_INPUT_FILE = Path(r'd:\Desktop\Data-Cleaner\Data\gun\BoltAction22_Samsung\SA_004A_S01.wav')

# Output (all modes write here)
OUTPUT_DIR = PROJECT_ROOT / 'Data' / 'TRIMMED_REALTIME'
```

### Changing Microphone Settings

```python
# For MICROPHONE mode:
MIC_DEVICE_INDEX = None          # None = default mic. Set to a number for specific device.
MIC_BUFFER_SECONDS = 2.0         # Rolling buffer length
MIC_TRIGGER_RMS = 0.05           # RMS threshold to trigger capture
MIC_CAPTURE_DURATION_S = 10.0    # How long to listen (seconds). Set to 0 for unlimited.
```

---

## 5. Microphone Setup (Windows)

### Step 1: Find Your Microphone Device Index

Run this in a Python cell:
```python
import sounddevice as sd
print(sd.query_devices())
```

You'll see output like:
```
  0 Microsoft Sound Mapper - Input, MME (2 in, 0 out)
  1 Microphone (Realtek Audio), MME (2 in, 0 out)
  2 Microsoft Sound Mapper - Output, MME (0 in, 2 out)
  3 Speakers (Realtek Audio), MME (0 in, 2 out)
> 4 Microphone (Realtek Audio), Windows DirectSound (2 in, 0 out)
```

Set `MIC_DEVICE_INDEX` to the number of your microphone (look for "Input" or "Microphone"):
```python
MIC_DEVICE_INDEX = 1  # Your microphone number
```

### Step 2: Check It Works

The notebook has a "Test Microphone" cell that records 2 seconds and plays it back.

### Step 3: Adjust Sensitivity

- **Too many false triggers**: Raise `MIC_TRIGGER_RMS` (e.g., from 0.05 to 0.1)
- **Missing events**: Lower `MIC_TRIGGER_RMS` (e.g., from 0.05 to 0.02)

---

## 6. Running The Notebook

| Cell | What It Does |
|------|-------------|
| 1 | Install dependencies |
| 2 | Imports |
| 3 | Configuration (set MODE + paths here) |
| 4 | Helper functions |
| 5 | Heuristic classifier |
| 6 | **Batch mode** — runs if `MODE == "batch"` |
| 7 | **Single file mode** — runs if `MODE == "single"` |
| 8 | **Microphone mode** — runs if `MODE == "microphone"` |
| 9 | Results summary |

> **Important**: Only the cell matching your selected MODE will execute. The others skip automatically.

---

## 7. Output Structure

```
Data/TRIMMED_REALTIME/
├── batch_output/
│   ├── gunshot/              ← Clips classified as gunshot
│   ├── nongunshot/           ← Clips classified as non-gunshot
│   └── uncertain/            ← Clips the classifier wasn't sure about
│
├── single_output/
│   ├── clips/                ← Extracted clips from the single file
│   └── analysis.png          ← Waveform + spectrogram + annotations
│
├── mic_captures/
│   ├── event_001_20260605_034512.wav
│   ├── event_002_20260605_034518.wav
│   └── ...                   ← Timestamped captures
│
└── reports/
    ├── manifest.csv
    └── summary.json
```

---

## 8. Troubleshooting

### "No module named 'sounddevice'"
```
pip install sounddevice
```
On Windows, this should work directly. Restart Jupyter after installing.

### "PortAudio library not found"
This happens on Linux/Mac. Install PortAudio:
```bash
# Ubuntu/Debian
sudo apt-get install libportaudio2

# macOS
brew install portaudio
```

### Microphone captures only silence
- Check your microphone is working in Windows Sound Settings
- Run the "Test Microphone" cell to verify
- Lower `MIC_TRIGGER_RMS` to be more sensitive

### Batch mode is slow
Processing thousands of files with full onset detection takes time. To speed up:
- Reduce the input folder to just the files you need
- Increase `BATCH_HOP_MS` for fewer windows per file

### Classification seems wrong
The heuristic classifier uses simple audio features — it's NOT a trained model. It's designed to be a rough first-pass for organizing test data. For accurate classification, train a proper ML model using Trimmers 01 and 02 output.

---

> **Remember**: This trimmer is for test data extraction and live capture. Its classifications are heuristic-based (good enough for organizing data), not model-based. After you train your ML/DL model, you can swap in the trained model for accurate predictions.
