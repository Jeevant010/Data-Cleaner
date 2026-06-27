# 🎯 Updated Trimmer Suite — Configurable Duration Pipeline

> **Date**: 2026-06-27  
> **Purpose**: Drop-in replacement for the original `Trimmer/` suite with a **configurable clip duration** (no longer locked to 250ms).

---

## Why This Exists

Our original Trimmer hardcoded 250ms clips. Academic research proves this is **dangerously short** for gunshot classification:

| Source | Classification Window Used | Notes |
|--------|---------------------------|-------|
| **gabemagee/gunshot_detection** (IEEE 2019) | **2.0 seconds** | Raspberry Pi deployment, 1D+2D CNN ensemble, 60k samples |
| **Saha et al. (2025)** — Automated Gunshot Detection in Forest Environments | **0.5 – 1.0 seconds** | Spectrogram + CNN, tested on real-world soundscapes |
| **Edge Impulse Public Projects** | **1.0 seconds** | Arduino Nano 33 BLE deployed, MFE/MFCC feature blocks |
| **arXiv: Exploring Feature Extraction Technique Parameters** (2026) | Varies, optimal at **0.5 – 2.0s** | Systematic study showing parameter tuning improves accuracy by up to 20% |

**Our 250ms window was cutting off the gunshot's reverb tail**, which is the primary feature that distinguishes gunshots from door slams, claps, and balloon pops.

---

## The Three Trimmers

| # | Trimmer | What It Does | Output Folder |
|---|---------|-------------|---------------|
| **01** | [Gunshot Trimmer](01_Gunshot_Trimmer/) | Extracts **verified real gunshot** clips with strict quality gates | `Data/TRIMMED_GUNSHOTS_{TARGET_MS}MS/` |
| **02** | [Non-Gunshot Trimmer](02_NonGunshot_Trimmer/) | Extracts **clean background** audio with impulse rejection | `Data/TRIMMED_NONGUNSHOTS_{TARGET_MS}MS/` |
| **03** | [Real-Time Trimmer](03_RealTime_Trimmer/) | Multi-mode test data processor: batch or single file | `Data/TRIMMED_REALTIME_{TARGET_MS}MS/` |

> **Dynamic folder naming**: If you set `TARGET_MS = 1000`, output goes to `TRIMMED_GUNSHOTS_1000MS/`. If you set `TARGET_MS = 500`, it goes to `TRIMMED_GUNSHOTS_500MS/`. Old 250ms data is **never overwritten**.

---

## Key Upgrade: Configurable Duration

Every notebook has a single config cell at the top:

```python
# ======= CHANGE THIS TO YOUR DESIRED CLIP DURATION =======
TARGET_MS = 1000    # Options: 250, 500, 750, 1000, 1500, 2000
# ==========================================================
```

Everything else (sample counts, pre-event padding, output folder names) auto-calculates from this single variable.

---

## Shared Specifications

| Parameter | Value |
|-----------|-------|
| Sample Rate | 22,050 Hz |
| Clip Duration | **Configurable** (default: 1000ms) |
| Channels | Mono |
| Bit Depth | 16-bit PCM WAV |
| Normalization | Peak-normalized to 0.99 |

---

## Recommended Settings by Use Case

| Use Case | Recommended `TARGET_MS` | Reasoning |
|----------|------------------------|-----------|
| **Arduino Nano 33 BLE** | 500 – 1000 | Must fit in 256KB RAM; 1000ms is the sweet spot |
| **Raspberry Pi** | 1000 – 2000 | More RAM available; longer windows = better accuracy |
| **PC-based detection** | 1000 – 2000 | No memory constraints |
| **Quick prototyping** | 500 | Faster training, more samples per file |

---

## Paper References & Related Work

### Academic Papers
1. **Magee et al. (2019)** — *"Low Cost Gunshot Detection using Deep Learning on the Raspberry Pi"*, IEEE International Conference on Big Data. Deployed on RPi 3B+ with 2-second buffers and 1D+2D CNN ensemble.
2. **Saha et al. (2025)** — *"Comparative Analysis of Deep Learning Architectures and Data Augmentation Strategies for Automated Gunshot Detection in Forest Environments"*, HuggingFace/arXiv.
3. **arXiv (2026)** — *"Exploring Feature Extraction Technique Parameters for Acoustic Gunshot Classification"*, systematic study of MFCC/MFE parameter optimization.

### GitHub Repositories
- [`gabemagee/gunshot_detection`](https://github.com/gabemagee/gunshot_detection) — RPi-based, IEEE-published, 2-second buffers, SMS alerts
- [`mariamkhmahran/gunshot-detection-system`](https://github.com/mariamkhmahran/gunshot-detection-system) — Urban 2D CNN + Mel-spectrogram
- [`hasnainnaeem/Gunshot-Detection-in-Audio`](https://github.com/hasnainnaeem/Gunshot-Detection-in-Audio) — TF 2.0, UrbanSound8K, binary classification
- [`sayandeepmaity/luminator`](https://github.com/sayandeepmaity/luminator) — Mic arrays + FPGA + hybrid CNN-RNN + 3D localization

### Arduino Nano 33 BLE — Proof of Deployment
- **Edge Impulse Public Project**: [`Gunshot Detection (ID: 133765)`](https://studio.edgeimpulse.com/public/133765/latest) — Working model deployed to Arduino Nano 33 BLE Sense via Edge Impulse's Arduino Library export.
- **"Go Ahead, Give AI a Shot"** — Edge Impulse blog post documenting full pipeline from data collection to BLE-based alerting on Nano 33 BLE hardware.
- **TinyML Audio Classification** — Official Edge Impulse + Arduino tutorial covering the exact MP34DT05 microphone → MFE → CNN → deployment pipeline.

> **Verdict:** People **have** successfully deployed gunshot detection on Arduino Nano 33 BLE via Edge Impulse. However, all successful deployments use **Edge Impulse's auto-optimized pipeline** (not raw TensorFlow). The models are typically tiny (~10-30KB) and use MFE features, not raw waveforms. Our custom 92K-parameter 1D CNN is too large for direct Arduino deployment without aggressive quantization or model shrinking.

---

## Quick Start

1. Open the notebook for the trimmer you want
2. **Set `TARGET_MS`** in Cell 3 to your desired duration
3. Run all cells
4. Check the dynamically-named output folder
