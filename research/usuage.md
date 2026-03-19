
# Gunshot-Focused Video Trimming Pipeline

This notebook builds a practical, modular pipeline to:
1. Extract audio from each video
2. Detect candidate gunshot moments from audio
3. Validate events with visual cues (flash + motion spikes)
4. Export trimmed video clips around confirmed events

Use this on your dataset folder once paths are set below.













## Notes For Better Accuracy

- Tune `zscore_threshold`, `flash_zscore_threshold`, and `motion_zscore_threshold` based on your data.
- For noisy environments, increase `zscore_threshold` or require both flash and motion by changing the validation condition.
- You can later replace the heuristic detector with a trained classifier while keeping the same pipeline structure.