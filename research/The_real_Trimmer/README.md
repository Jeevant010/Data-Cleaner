# The Real Trimmer (Strict 250 ms)

This folder contains a strict dataset builder that creates fixed 250 ms clips with clean label separation.

## What it builds

- `class_1_gunshot`: clips extracted from `Data/gun` and `Data/edge-collected-gunshot-audio`
- `class_0_nongunshot`: clips extracted from `Data/sound` (optionally `Data/audio`)
- `noise`: uncertain clips + silence + generated pitch-zero clips

All output clips have identical size (`target_ms`) and sample rate.

## Output location

Running the script creates:

- `Data/TRIMMED_250MS_STRICT/class_1_gunshot`
- `Data/TRIMMED_250MS_STRICT/class_0_nongunshot`
- `Data/TRIMMED_250MS_STRICT/noise`
- `Data/TRIMMED_250MS_STRICT/reports`

Reports:

- `manifest.csv`: every produced clip with metadata
- `rejected_sources.csv`: unreadable/invalid source files
- `summary.json`: final build counts and config

## Run

From project root:

```powershell
python research/The_real_Trimmer/build_clean_250ms_dataset.py --overwrite
```

Optional flags:

- `--include-audio-folder` to include `Data/audio` in class 0 generation
- `--class0-ratio 1.2` to generate 20% more class 0 than class 1
- `--zero-noise-count 500` to generate more pitch-zero clips
- `--target-ms 250 --sample-rate 22050` to control final clip size

## Important behavior

- Corrupt or empty files are skipped and logged.
- Gunshot clips are selected around transient candidates and must pass strict checks.
- Non-gunshot windows with impulsive signatures are diverted into `noise/uncertain_from_class0`.
- `noise/pitch_zero` contains pure silence clips (exact zero signal).
