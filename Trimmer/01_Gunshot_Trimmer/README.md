# 🔫 01 — Gunshot Trimmer

Extracts the **maximum number of verified, real gunshot clips** from all gunshot source data. Zero tolerance for false positives.

## What It Does

- Scans `Data/gun/` and `Data/edge-collected-gunshot-audio/`
- Uses onset detection to find every genuine gunshot event
- Extracts up to **8 clips per source file** (only real onsets, never pads with silence)
- Applies strict quality gates: peak, RMS, crest factor, prominence, spectral centroid
- Every output clip is **exactly 250ms** (5,512 samples @ 22,050 Hz)

## Output

```
Data/TRIMMED_GUNSHOTS/
├── verified/           ← All confirmed gunshot clips
├── rejected/           ← Clips that failed quality checks (for manual review)
└── reports/
    ├── manifest.csv    ← Every clip with metadata
    └── summary.json    ← Build stats
```

## Run

1. Open `gunshot_trimmer.ipynb` in Jupyter or VS Code
2. Run all cells in order
3. Check the verification cell output

## Full Documentation

See [manual/README.md](manual/README.md) for the complete manual including path configuration, quality gate explanations, and troubleshooting.
