# Audio Concepts for Beginners

> This document explains the audio and signal processing concepts used in the pipeline. No prior knowledge assumed.

---

## What is a WAV File?

A WAV file stores sound as a series of numbers. Each number represents the air pressure at one tiny moment in time.

```
Time:     0.0s    0.001s   0.002s   0.003s   ...
Value:    0.00    0.12    -0.05     0.31    ...
```

- The **sample rate** tells you how many of these numbers are recorded per second
- At 22,050 Hz, there are 22,050 numbers per second
- So 250 milliseconds = 22,050 × 0.25 = **5,512 numbers (samples)**

## What is a Sample Rate?

Think of it like frames per second in a video:
- A video at 30 FPS takes 30 pictures per second
- Audio at 22,050 Hz takes 22,050 "pictures of air pressure" per second

Higher sample rate = more detail, but bigger files. 22,050 Hz is plenty for detecting gunshots.

## What is Mono vs Stereo?

- **Stereo**: Two channels (left ear, right ear). Takes twice as much space.
- **Mono**: One channel. We use this because we don't care which direction the gunshot came from — just whether it happened.

## What is RMS (Root Mean Square)?

RMS is the "average loudness" of an audio clip. It's calculated by:
1. Square every sample value (makes negatives positive)
2. Take the mean (average)
3. Take the square root

```python
rms = sqrt(mean(samples²))
```

A gunshot has high RMS because it's loud. Pure silence has RMS = 0.

## What is Peak?

The single loudest sample in the clip. A gunshot has a very high peak because the initial "bang" is extremely loud.

## What is Crest Factor?

```
Crest Factor = Peak / RMS
```

A **high crest factor** means the sound has one very loud moment but is quiet on average. Gunshots have extremely high crest factors (the "bang" is loud, but the rest is quiet echo/silence).

A **low crest factor** means the sound is consistently loud (like a car engine or music).

We use the crest factor to tell the difference between:
- Gunshot (high crest = 10+): one big spike
- Background noise (low crest = 2-5): consistent sound

## What is an Onset?

An "onset" is the exact moment a sound begins. For a gunshot, the onset is the millisecond the bullet fires — the sudden jump from silence to maximum volume.

`librosa.onset.onset_detect()` finds these moments automatically by looking for sudden jumps in energy.

## What is a Transient?

A transient is a very short, sharp sound. Gunshots are the classic example:
- The actual "bang" lasts only 5-20 milliseconds
- Then there's 50-200ms of echo/decay
- Then silence

This is why cutting audio at fixed intervals fails — the transient can land anywhere and get split in half.

## What is the Mel Spectrogram?

A mel spectrogram is a visual representation of sound that shows:
- **X-axis**: Time (left to right)
- **Y-axis**: Frequency (low sounds at bottom, high sounds at top)
- **Color**: Loudness (darker = louder)

It's called "mel" because it uses a scale that matches how human ears perceive pitch. Low frequencies get more detail, high frequencies are compressed.

CNN models typically learn from mel spectrograms rather than raw audio because they capture the frequency patterns that distinguish a gunshot from other sounds.
