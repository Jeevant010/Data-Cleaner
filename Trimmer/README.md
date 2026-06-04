# 🎯 Trimmer Suite — Final Data Cleaning Pipeline

> **Purpose**: Three specialized audio trimmers that produce model-ready datasets for gunshot detection ML/DL training.

---

## The Three Trimmers

| # | Trimmer | What It Does | Output Folder |
|---|---------|-------------|---------------|
| **01** | [Gunshot Trimmer](01_Gunshot_Trimmer/) | Extracts **verified real gunshot** clips with strict quality gates | `Data/TRIMMED_GUNSHOTS/` |
| **02** | [Non-Gunshot Trimmer](02_NonGunshot_Trimmer/) | Extracts **clean background** audio with impulse rejection | `Data/TRIMMED_NONGUNSHOTS/` |
| **03** | [Real-Time Trimmer](03_RealTime_Trimmer/) | Multi-mode test data processor: batch, single file, or **live microphone** | `Data/TRIMMED_REALTIME/` |

---

## Recommended Workflow

```
Step 1 ─→ Run 01_Gunshot_Trimmer
           ↳ Produces 4,000+ verified gunshot clips (250ms each)

Step 2 ─→ Run 02_NonGunshot_Trimmer
           ↳ Produces balanced non-gunshot clips (250ms each)

Step 3 ─→ Run 03_RealTime_Trimmer (for test data)
           ↳ Feed it unseen audio → extracts test clips
           ↳ OR capture live audio from your microphone

Step 4 ─→ Train your ML/DL model
           ↳ Use Trimmer 01 + 02 output for train/val
           ↳ Use Trimmer 03 output for test/evaluation
```

---

## Data Sources

All trimmers read from the same raw data under `Data/`:

| Folder | Content | Used By |
|--------|---------|---------|
| `Data/gun/` | ~2,403 Samsung-recorded gunshot samples | Trimmer 01 |
| `Data/edge-collected-gunshot-audio/` | ~2,148 edge-collected gunshots | Trimmer 01 |
| `Data/sound/` | ~230 car/street/driving sounds | Trimmer 02 |
| `Data/audio/` | ~6,000 environmental sounds | Trimmer 02 |
| *Any folder you choose* | Your test audio files | Trimmer 03 |

---

## Shared Specifications

All trimmers produce clips with identical specs:

| Parameter | Value |
|-----------|-------|
| Sample Rate | 22,050 Hz |
| Clip Duration | 250 ms (5,512 samples) |
| Channels | Mono |
| Bit Depth | 16-bit PCM WAV |
| Normalization | Peak-normalized to 0.99 |

---

## Existing Data

> **Note**: The existing `Data/TRIMMED_250MS_STRICT/` dataset (produced by `research/The_real_Trimmer/`) is **NOT touched** by any of these trimmers. Each writes to its own separate output folder.

---

## Quick Start

1. Open the notebook for the trimmer you want in Jupyter / VS Code
2. Run Cell 1 (install dependencies)
3. Check Cell 3 (configuration — paths auto-detect, but verify)
4. Run all cells
5. Check the output folder

Each trimmer has its own `manual/README.md` with full documentation.
