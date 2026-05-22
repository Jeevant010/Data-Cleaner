# Configuration Parameters Reference

> All the settings you can change and what they do.

---

## Audio Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | `22050` | Target sample rate in Hz. All files are resampled to this. Use 22050 for a good balance of quality and file size. Use 16000 for TinyML if your microcontroller expects it. |
| `TARGET_MS` | `250` | Clip duration in milliseconds. Every single output file will be exactly this long. |
| `TARGET_SAMPLES` | `5512` | Automatically calculated: `SAMPLE_RATE × TARGET_MS / 1000`. Do not change manually. |

## Gunshot Extraction Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PRE_EVENT_MS` | `50` | Milliseconds of audio to grab BEFORE the detected onset. This captures the silence right before the bang. |
| `PRE_EVENT_SAMPLES` | `1102` | Automatically calculated. Do not change manually. |

### What Happens If You Change PRE_EVENT_MS?

| Value | Effect |
|-------|--------|
| `0` | Clip starts exactly at the bang. Risk: slight onset detection error could cut the very start of the gunshot. |
| `25` | Small safety margin. |
| `50` (default) | Good balance — captures pre-silence context and handles detection jitter. |
| `100` | More pre-silence, but less post-bang decay captured (only 150ms of decay instead of 200ms). |

## Background Extraction Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLASS0_HOP_MS` | `125` | Sliding window step size. 125ms = 50% overlap with the 250ms window. |
| `CLASS0_HOP_SAMPLES` | `2756` | Automatically calculated. Do not change manually. |

### What Happens If You Change CLASS0_HOP_MS?

| Value | Overlap | Clips Per Second | Effect |
|-------|---------|------------------|--------|
| `250` | 0% | 4 per second | No overlap. Fast but may miss transitions. |
| `125` (default) | 50% | 8 per second | Good coverage. Standard in audio ML. |
| `62` | 75% | 16 per second | Very dense coverage. Many more clips, much slower. |

## Source Directories

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLASS1_DIRS` | `[Data/gun, Data/edge-collected-gunshot-audio]` | Folders containing gunshot recordings |
| `CLASS0_DIRS` | `[Data/sound, Data/audio]` | Folders containing non-gunshot recordings |

To add more folders, just append them to the list:
```python
CLASS0_DIRS = [DATA_DIR / 'sound', DATA_DIR / 'audio', DATA_DIR / 'my_new_folder']
```

## Safety Thresholds

These are hardcoded in the extraction functions:

| Check | Value | Where Used | What It Does |
|-------|-------|------------|-------------|
| Crest factor > 8.5 AND peak > 0.1 | Background extraction | Rejects clips that look like gunshots |
| Peak < 0.003 | Background extraction | Rejects pure silence |
| Peak < 0.01 | Gunshot extraction | Rejects clips with no audible sound |
