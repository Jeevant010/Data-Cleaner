# Code Walkthrough — Line by Line

> This document explains every function in the notebook so you understand exactly what each piece of code does.

---

## 1. `load_audio(path)`

```python
def load_audio(path):
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.nan_to_num(y)
```

**What it does:**
- `librosa.load()` reads the WAV file and converts it to a numpy array
- `sr=SAMPLE_RATE` resamples to 22,050 Hz (even if the original file was 44,100 or 16,000 Hz)
- `mono=True` mixes stereo channels into one
- `np.nan_to_num()` replaces any corrupted values (NaN, infinity) with 0

**Returns:** A 1D numpy array of float32 values between -1.0 and 1.0

---

## 2. `force_exact_length(clip)`

```python
def force_exact_length(clip):
    if len(clip) == TARGET_SAMPLES:
        return clip
    if len(clip) > TARGET_SAMPLES:
        return clip[:TARGET_SAMPLES]
    return np.pad(clip, (0, TARGET_SAMPLES - len(clip)), mode='constant')
```

**What it does:**
This is the **iron guarantee** that every clip is exactly 5,512 samples. No matter what:
- If the clip is already the right length → return as-is
- If too long → chop off the end
- If too short → pad with zeros (silence)

**Why it matters:** This single function prevents the "different-sized clips" cheating problem.

---

## 3. `normalize_clip(clip)`

```python
def normalize_clip(clip):
    peak = np.max(np.abs(clip))
    if peak > 0.99:
        clip = (clip / peak) * 0.99
    return clip
```

**What it does:**
If any sample exceeds 0.99 (which would cause distortion when saved as 16-bit audio), it scales the entire clip down proportionally.

**Example:**
- Input: `[0.5, -1.5, 0.3]` (peak = 1.5, which is too high)
- Output: `[0.33, -0.99, 0.2]` (scaled so peak = 0.99)

---

## 4. `extract_gunshot_clips(y)` — The Sniper Method

```python
# Step 1: Find onsets
onset_frames = librosa.onset.onset_detect(y=y, sr=SAMPLE_RATE, ...)
onset_samples = librosa.frames_to_samples(onset_frames)
```

`onset_detect` returns frame indices (each frame = 512 samples by default). `frames_to_samples` converts these back to exact sample positions.

```python
# Step 2: Remove overlapping onsets
min_gap = TARGET_SAMPLES // 2
filtered = []
for onset in sorted(onset_samples):
    if not filtered or (onset - filtered[-1]) >= min_gap:
        filtered.append(onset)
```

If two onsets are closer than half a window (2,756 samples = 125ms), we keep only the first one. This prevents overlapping clips from the same burst.

```python
# Step 3: Cut the window around each onset
start = int(onset) - PRE_EVENT_SAMPLES  # 50ms before
end = start + TARGET_SAMPLES             # 250ms total
```

For each onset, we grab a window that starts 50ms before the bang and extends 200ms after it.

```python
# Step 4: Handle edge cases
pad_left = max(0, -start)   # If start is negative (onset near file start)
pad_right = max(0, end - len(y))  # If end exceeds file length
```

If the onset is within the first 50ms of the file, `start` would be negative. We pad with zeros instead of crashing.

---

## 5. `extract_background_clips(y)` — Sliding Window

```python
for start in range(0, len(y) - TARGET_SAMPLES + 1, CLASS0_HOP_SAMPLES):
    end = start + TARGET_SAMPLES
    clip = y[start:end]
```

This slides through the audio in steps of 125ms (2,756 samples), grabbing a 250ms window each time. The 50% overlap ensures we don't miss any part of the audio.

```python
# Safety check
peak = np.max(np.abs(clip))
rms = np.sqrt(np.mean(clip ** 2) + 1e-10)
crest = peak / rms

if crest > 8.5 and peak > 0.1:
    continue  # Skip — this might be a gunshot!
```

If a background clip has a very sharp peak (crest factor > 8.5) AND is reasonably loud (peak > 0.1), it's suspiciously gunshot-like and gets skipped. This prevents accidental gunshot sounds from contaminating the background class.

---

## 6. `build_dataset()` — The Main Pipeline

The pipeline runs in this order:

```
1. Collect all source files
2. Extract ALL Class 1 clips (no limit)
3. Extract ALL Class 0 clips (no limit)
4. Count both → take the minimum
5. Randomly subsample both to match
6. Write to disk
7. Save manifest CSV and summary JSON
```

Key design decision: We extract **everything first**, then balance. This ensures we use the maximum possible amount of data while maintaining perfect 1:1 balance.

---

## Configuration Parameters Quick Reference

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `SAMPLE_RATE` | 22,050 | All audio resampled to this Hz |
| `TARGET_MS` | 250 | Every clip is exactly this many milliseconds |
| `TARGET_SAMPLES` | 5,512 | = SAMPLE_RATE × TARGET_MS / 1000 |
| `PRE_EVENT_MS` | 50 | How far before the gunshot onset to start the clip |
| `CLASS0_HOP_MS` | 125 | Sliding window step size (50% of 250ms) |
