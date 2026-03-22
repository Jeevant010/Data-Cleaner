
# Gunshot-Focused Video Trimming Pipeline

This notebook builds a practical, modular pipeline to:
1. Extract audio from each video
2. Detect candidate gunshot moments from audio
3. Validate events with visual cues (flash + motion spikes)
4. Export trimmed video clips around confirmed events

## How to Use the Notebook (`gunshot_event_trimming_pipeline.ipynb`)

1. **Prerequisites (FFmpeg Setup for Windows)**: The pipeline requires FFmpeg to extract audio and trim videos. The easiest way to install the pre-compiled version on Windows without manually setting PATH variables is via `winget`:
   - Open your Command Prompt or PowerShell.
   - Run the following command:
     ```powershell
     winget install ffmpeg
     ```
   - **Important**: Restart your Jupyter notebook, VSCode, or Command Prompt completely so it recognizes the newly installed `ffmpeg` command.
   *(Note: Downloading directly from ffmpeg.org often gives you the uncompiled source code. Using `winget` automatically fetches and installs the correct `.exe` files for Windows).*
2. **Install Packages**: Run the very first cell to install Python dependencies (`numpy`, `pandas`, `librosa`, `opencv-python`, `tqdm`).
3. **Set Up Paths (Using the `Data` folder)**: Go to the **config** cell and update the `CONFIG` object to point to the `Data` folder that was added:
   - Change `dataset_root` to `Path(r'c:\Desktop\Data-Cleaner\research\Data')`. This tells the notebook where your raw files are.
   - Change `output_root` to `Path(r'c:\Desktop\Data-Cleaner\research\Data\Output')`. The pipeline will automatically create this `Output` folder and organize your clips and CSV report there.
4. **Execute**: Run all cells in sequence. The final cell calls `run_dataset_pipeline(CONFIG)`, processes your dataset, and generates the clips.

Use this on your dataset folder once paths are set below.








## Notes For Better Accuracy

- Tune `zscore_threshold`, `flash_zscore_threshold`, and `motion_zscore_threshold` based on your data.
- For noisy environments, increase `zscore_threshold` or require both flash and motion by changing the validation condition.
- You can later replace the heuristic detector with a trained classifier while keeping the same pipeline structure.