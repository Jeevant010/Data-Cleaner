# 🎵 02 — Non-Gunshot Trimmer

Extracts **clean background audio clips** with strict impulse rejection to ensure no gunshot-like sounds leak into the non-gunshot class.

## What It Does

- Scans `Data/sound/` and optionally `Data/audio/`
- Uses sliding window with 50% overlap (125ms hop) to extract 250ms clips
- **Impulse rejection**: Rejects any window that looks like a gunshot (high crest + high peak)
- **Silence rejection**: Skips pure silence windows
- **Diversity sampling**: Caps clips per file, ensures variety across sources
- Every output clip is **exactly 250ms** (5,512 samples @ 22,050 Hz)

## Output

```
Data/TRIMMED_NONGUNSHOTS/
├── verified/           ← All confirmed non-gunshot clips
├── suspicious/         ← Clips rejected by impulse filter (for review)
└── reports/
    ├── manifest.csv
    └── summary.json
```

## Run

1. Open `nongunshot_trimmer.ipynb` in Jupyter or VS Code
2. Run all cells in order
3. Check the verification cell output

## Full Documentation

See [manual/README.md](manual/README.md) for the complete manual.
