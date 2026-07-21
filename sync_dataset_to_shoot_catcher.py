"""
===============================================================================
Data-Cleaner to Shoot_Catcher — Intelligent Automated Dataset Sync Pipeline
===============================================================================
- Automatically collects trimmed audio from Data-Cleaner output folders.
- Applies Source-Group Stratified Sampling on non-gunshot clips to maintain
  an optimal, balanced ratio (~1:12) with maximum acoustic source diversity.
- Automatically populates Shoot_Catcher/Data/READY_1D_CNN and READY_2D_CNN.
===============================================================================
"""

import os
import sys
import io
import re
import random
import shutil
import json
from pathlib import Path
from collections import defaultdict

# Fix Windows console encoding for print output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# --- CONFIGURATION ---
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent

# Data-Cleaner Source Folders
DATA_CLEANER_DIR = SCRIPT_DIR
GUNSHOT_SEARCH_DIRS = [
    DATA_CLEANER_DIR / 'Data' / 'TRIMMED_GUNSHOTS_750MS' / 'verified',
    DATA_CLEANER_DIR / 'Data' / 'TRIMMED_GUNSHOTS' / 'verified',
    DATA_CLEANER_DIR / 'Data' / 'TRIMMED_GUNSHOTS_750MS',
    DATA_CLEANER_DIR / 'Data' / 'TRIMMED_GUNSHOTS',
]

NONGUNSHOT_SEARCH_DIRS = [
    DATA_CLEANER_DIR / 'Data' / 'TRIMMED_NONGUNSHOTS_750MS' / 'clean',
    DATA_CLEANER_DIR / 'Data' / 'TRIMMED_NONGUNSHOTS' / 'clean',
    DATA_CLEANER_DIR / 'Data' / 'TRIMMED_NONGUNSHOTS_750MS',
    DATA_CLEANER_DIR / 'Data' / 'TRIMMED_NONGUNSHOTS',
]

# Shoot_Catcher Target Folders
SHOOT_CATCHER_DATA = PROJECT_ROOT / 'Shoot_Catcher' / 'Data'
TARGET_1D_DIR = SHOOT_CATCHER_DATA / 'READY_1D_CNN'
TARGET_2D_DIR = SHOOT_CATCHER_DATA / 'READY_2D_CNN'

TARGET_RATIO = 12.0  # Target ratio of Non-Gunshots to Gunshots (1:12)
SEED = 42

random.seed(SEED)

def extract_source_group(filename):
    """Extract root audio source name to prevent source leakage and allow stratified sampling."""
    stem = Path(filename).stem
    no_prefix = re.sub(r'^c[01]_\d+_', '', stem)
    no_suffix = re.sub(r'_clip\d+.*$|_onset\d+.*$', '', no_prefix)
    no_aug = re.sub(r'_aug_\d+dB$', '', no_suffix)
    return no_aug if no_aug else stem


def find_audio_files(search_dirs):
    """Find all .wav files in given list of search directories."""
    files = []
    for d in search_dirs:
        if d.exists():
            wavs = list(d.rglob('*.wav'))
            if wavs:
                print(f"  Found {len(wavs):,} WAV files in: {d.relative_to(PROJECT_ROOT)}")
                files.extend(wavs)
                break  # Pick the first matching prioritized directory
    return files


def main():
    print("=" * 80)
    print("🔄 INTELLIGENT DATASET SYNC PIPELINE (Data-Cleaner ➔ Shoot_Catcher)")
    print("=" * 80)

    print("\n🔍 Locating Trimmed Audio Files in Data-Cleaner...")
    gunshot_files = find_audio_files(GUNSHOT_SEARCH_DIRS)
    nongunshot_files = find_audio_files(NONGUNSHOT_SEARCH_DIRS)

    if not gunshot_files:
        print("❌ Error: No gunshot WAV files found in Data-Cleaner output folders!")
        print("   Please run gunshot_trimmer.ipynb first.")
        return

    if not nongunshot_files:
        print("❌ Error: No non-gunshot WAV files found in Data-Cleaner output folders!")
        print("   Please run nongunshot_trimmer.ipynb first.")
        return

    n_gunshots = len(gunshot_files)
    target_nongunshots = int(n_gunshots * TARGET_RATIO)
    print(f"\n📊 Gunshots Available      : {n_gunshots:,}")
    print(f"📊 Non-Gunshots Available  : {len(nongunshot_files):,}")
    print(f"🎯 Target Ratio (1:{TARGET_RATIO:.1f})   : {target_nongunshots:,} Non-Gunshots")

    # --- Stratified Sub-Sampling by Source Group ---
    print("\n🧠 Grouping Non-Gunshots by Source Audio to Maximize Diversity...")
    source_groups = defaultdict(list)
    for f in nongunshot_files:
        grp = extract_source_group(f.name)
        source_groups[grp].append(f)

    n_groups = len(source_groups)
    print(f"   Found {n_groups:,} unique background audio source groups.")

    selected_nongunshots = []
    if len(nongunshot_files) <= target_nongunshots:
        print("   Total non-gunshots is less than or equal to target ratio. Including all files.")
        selected_nongunshots = nongunshot_files.copy()
    else:
        # Round-robin sampling across all source groups for max acoustic diversity
        group_keys = list(source_groups.keys())
        random.shuffle(group_keys)

        per_group_quota = max(1, target_nongunshots // n_groups)
        print(f"   Sampling ~{per_group_quota} clip(s) per source group...")

        # Phase 1: Equal quota per group
        for grp in group_keys:
            clips = source_groups[grp]
            sampled = random.sample(clips, min(len(clips), per_group_quota))
            selected_nongunshots.extend(sampled)

        # Phase 2: Fill remaining if needed
        remaining_needed = target_nongunshots - len(selected_nongunshots)
        if remaining_needed > 0:
            already_selected_set = set(selected_nongunshots)
            pool = [f for f in nongunshot_files if f not in already_selected_set]
            if pool:
                additional = random.sample(pool, min(len(pool), remaining_needed))
                selected_nongunshots.extend(additional)

    random.shuffle(selected_nongunshots)
    actual_ratio = len(selected_nongunshots) / n_gunshots
    print(f"\n✅ Final Selection: {n_gunshots:,} Gunshots vs {len(selected_nongunshots):,} Non-Gunshots (Ratio 1:{actual_ratio:.1f})")

    # --- Copying to Shoot_Catcher Target Folders ---
    for target_root in [TARGET_1D_DIR, TARGET_2D_DIR]:
        print(f"\n📂 Preparing Target Directory: {target_root.relative_to(PROJECT_ROOT)}...")
        c1_dir = target_root / 'class_1_gunshot'
        c0_dir = target_root / 'class_0_nongunshot'

        # Clear and recreate clean target folders
        shutil.rmtree(c1_dir, ignore_errors=True)
        shutil.rmtree(c0_dir, ignore_errors=True)
        c1_dir.mkdir(parents=True, exist_ok=True)
        c0_dir.mkdir(parents=True, exist_ok=True)

        print("  Copying class_1_gunshot files...")
        for f in gunshot_files:
            shutil.copy2(f, c1_dir / f.name)

        print("  Copying class_0_nongunshot files...")
        for f in selected_nongunshots:
            shutil.copy2(f, c0_dir / f.name)

    # Save summary metadata
    summary = {
        'total_gunshots': len(gunshot_files),
        'total_nongunshots': len(selected_nongunshots),
        'ratio': f"1:{actual_ratio:.2f}",
        'unique_background_sources': n_groups,
        'target_1d': str(TARGET_1D_DIR),
        'target_2d': str(TARGET_2D_DIR),
    }
    (SHOOT_CATCHER_DATA / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print("🎉 DATASET SYNC COMPLETE!")
    print(f"   READY_1D_CNN: {len(gunshot_files):,} Gunshots | {len(selected_nongunshots):,} Non-Gunshots")
    print(f"   READY_2D_CNN: {len(gunshot_files):,} Gunshots | {len(selected_nongunshots):,} Non-Gunshots")
    print("   Metadata saved to: Shoot_Catcher/Data/dataset_summary.json")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
