# 🎙️ 03 — Real-Time / Test Data Trimmer

A **multi-mode trimmer** for extracting test data. Switch between three modes:

| Mode | Description |
|------|------------|
| `batch` | Feed it a folder of audio files → extracts and classifies all clips |
| `single` | Give it one audio file → detailed analysis + clip extraction |
| `microphone` | Capture live audio from your PC's mic → saves detected events |

## What It Does

- Processes audio through the same quality pipeline as Trimmers 01 & 02
- Uses energy-based onset detection (no ML model needed)
- Classifies clips using audio features: energy, crest factor, ZCR, spectral centroid
- Every output clip is **exactly 250ms** (5,512 samples @ 22,050 Hz)
- **You choose the mode** by setting one variable in the config cell

## Output

```
Data/TRIMMED_REALTIME/
├── batch_output/
│   ├── gunshot/         ← Detected gunshot clips
│   └── nongunshot/      ← Detected non-gunshot clips
├── single_output/       ← Single file analysis results
├── mic_captures/        ← Live microphone captures
└── reports/
    ├── manifest.csv
    └── summary.json
```

## Run

1. Open `realtime_trimmer.ipynb` in Jupyter or VS Code
2. Set `MODE = "batch"` or `"single"` or `"microphone"` in the config cell
3. Run all cells in order

## Full Documentation

See [manual/README.md](manual/README.md) for the complete manual including microphone setup on Windows.
