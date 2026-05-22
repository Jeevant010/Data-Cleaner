# Why Dataset Balance Matters

> This document explains why having equal numbers of gunshot and non-gunshot clips is critical for your model.

---

## The Cheating Problem

Imagine you're a student taking a true/false test where 99% of the answers are "False".

If you just write "False" for every single question — without even reading them — you'll score 99%. You didn't learn anything, but you got a great score by exploiting the pattern.

**This is exactly what happens to a machine learning model with imbalanced data.**

## Real Example From This Project

In the previous pipeline run, the output was:
- Class 1 (Gunshots): **34,419 clips**
- Class 0 (Background): **8,950 clips**

If a model just learned to always say "Gunshot", it would be right 79% of the time (34,419 out of 43,369 total). That's a terrible model that thinks everything is a gunshot — but it has "79% accuracy".

## How Balance Fixes This

When you have exactly the same number of clips in both classes:
- Class 1 (Gunshots): **10,000 clips**
- Class 0 (Background): **10,000 clips**

Now if the model always guesses one class, it only gets 50% accuracy. To get higher accuracy, it **must actually learn the difference** between a gunshot sound and a background sound.

## Why All Clips Must Be The Same Size

Another way a model can cheat is through file size/length:

| Scenario | What The Model Learns |
|----------|----------------------|
| Gunshots = 100ms, Background = 500ms | "If the clip is short, say gunshot" |
| Gunshots = 250ms, Background = 250ms | "I need to actually listen to the sound" |

By making every single clip exactly 250ms (5,512 samples), we force the model to learn from the **content** of the audio, not the **container**.

## The Pipeline's Balancing Strategy

```
Step 1: Extract ALL gunshot clips     → 10,000 clips (example)
Step 2: Extract ALL background clips  → 50,000 clips (example)
Step 3: min(10000, 50000) = 10,000
Step 4: Randomly pick 10,000 from each class
Step 5: Write 10,000 + 10,000 = 20,000 total clips
```

This means you always get the **maximum possible balanced dataset**. No data is wasted from the minority class.

## What About "Augmentation" To Fix Imbalance?

Sometimes people generate fake data to balance an imbalanced dataset. For example, pitch-shifting a gunshot to create a "new" gunshot clip. 

**This is dangerous for transient sounds like gunshots.** Pitch-shifting stretches or compresses the waveform, making the gunshot sound like a drum or a thud. The model then learns what a "fake, stretched gunshot" sounds like instead of a real one.

The better approach (which this pipeline uses) is:
1. Extract all real clips
2. Downsample the larger class to match the smaller one
3. If you want more data, collect more real recordings
