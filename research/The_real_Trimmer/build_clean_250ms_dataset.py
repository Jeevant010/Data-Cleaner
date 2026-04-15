"""
Create a strict 250 ms gunshot dataset with clean class separation.

Class 1 sources:
  - Data/gun
  - Data/edge-collected-gunshot-audio

Class 0 source:
  - Data/sound (default)
  - Data/audio (optional via --include-audio-folder)

Outputs under Data/<output_name>/:
  - class_1_gunshot/
  - class_0_nongunshot/
  - noise/
      - uncertain_from_class1/
      - uncertain_from_class0/
      - silence/
      - pitch_zero/
  - reports/
      - manifest.csv
      - rejected_sources.csv
      - summary.json

The pipeline enforces fixed clip size and skips/writes out corrupted or empty clips.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm


JUNK_TOKENS = ("__MACOSX",)
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ClipMetrics:
    rms: float
    peak: float
    crest_factor: float
    attack_ratio: float
    median_abs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict 250 ms audio dataset.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root path that contains Data/. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="TRIMMED_250MS_STRICT",
        help="Folder name to create inside Data/.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate for all output clips.",
    )
    parser.add_argument(
        "--target-ms",
        type=int,
        default=250,
        help="Fixed clip duration in milliseconds.",
    )
    parser.add_argument(
        "--class0-ratio",
        type=float,
        default=1.0,
        help="Target class0 clips relative to class1 clips.",
    )
    parser.add_argument(
        "--max-class1-per-file",
        type=int,
        default=1,
        help="Maximum number of class1 clips to keep per source file.",
    )
    parser.add_argument(
        "--include-audio-folder",
        action="store_true",
        help="Include Data/audio as extra class0 source.",
    )
    parser.add_argument(
        "--zero-noise-count",
        type=int,
        default=250,
        help="How many pure-silence (pitch-zero) clips to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output folder before writing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def find_project_root(start: Path) -> Path:
    expected = ("Data",)
    for candidate in [start.resolve(), *start.resolve().parents]:
        if all((candidate / token).exists() for token in expected):
            return candidate
    raise FileNotFoundError("Could not locate project root containing Data/.")


def is_junk_file(path: Path) -> bool:
    as_posix = path.as_posix()
    if any(token in as_posix for token in JUNK_TOKENS):
        return True
    if path.name.startswith("._"):
        return True
    return False


def sanitize_name(raw: str, max_len: int = 120) -> str:
    cleaned = SAFE_NAME_RE.sub("_", raw.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = "clip"
    return cleaned[:max_len]


def collect_wavs(source_dirs: Iterable[Path], source_label: str) -> list[dict]:
    records: list[dict] = []
    for root in source_dirs:
        if not root.exists():
            continue
        for wav in root.rglob("*.wav"):
            if is_junk_file(wav):
                continue
            records.append(
                {
                    "path": wav,
                    "source_group": source_label,
                    "source_dir": root.name,
                }
            )
    records.sort(key=lambda x: str(x["path"]))
    return records


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32)


def compute_metrics(clip: np.ndarray) -> ClipMetrics:
    abs_clip = np.abs(clip)
    peak = float(np.max(abs_clip)) if len(abs_clip) else 0.0
    rms = float(np.sqrt(np.mean(np.square(clip)))) if len(clip) else 0.0
    crest = peak / (rms + 1e-12)
    attack = float(np.max(np.abs(np.diff(clip)))) if len(clip) > 1 else 0.0
    attack_ratio = attack / (rms + 1e-12)
    med = float(np.median(abs_clip)) if len(abs_clip) else 0.0
    return ClipMetrics(
        rms=rms,
        peak=peak,
        crest_factor=crest,
        attack_ratio=attack_ratio,
        median_abs=med,
    )


def get_fixed_window(
    y: np.ndarray,
    center_sample: int,
    target_len: int,
    pre_event_samples: int,
) -> tuple[np.ndarray, int, int]:
    start = center_sample - pre_event_samples
    end = start + target_len

    pad_left = max(0, -start)
    pad_right = max(0, end - len(y))

    src_start = max(0, start)
    src_end = min(len(y), end)
    clip = y[src_start:src_end]

    if pad_left or pad_right:
        clip = np.pad(clip, (pad_left, pad_right))

    if len(clip) != target_len:
        clip = librosa.util.fix_length(clip, size=target_len)

    return clip.astype(np.float32), src_start, src_end


def get_class1_candidate_centers(y: np.ndarray, sample_rate: int, target_len: int) -> list[int]:
    if len(y) == 0:
        return []

    abs_y = np.abs(y)
    smooth_kernel = max(3, int(0.005 * sample_rate))
    kernel = np.ones(smooth_kernel, dtype=np.float32) / smooth_kernel
    envelope = np.convolve(abs_y, kernel, mode="same")

    peak_raw = int(np.argmax(abs_y))
    peak_env = int(np.argmax(envelope))

    hop = 128
    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop,
        backtrack=False,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=3,
        delta=0.1,
        wait=2,
    )

    candidates = [peak_raw, peak_env]
    candidates.extend(int(frame * hop) for frame in onset_frames)

    scores = {}
    diff = np.abs(np.diff(y, prepend=0.0))
    for c in candidates:
        if 0 <= c < len(y):
            scores[c] = float(abs_y[c]) + float(diff[c])

    sorted_candidates = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    # Keep centers separated so multi-shot files can contribute distinct events.
    min_gap = max(target_len // 2, int(sample_rate * 0.03))
    filtered: list[int] = []
    for c in sorted_candidates:
        if all(abs(c - kept) >= min_gap for kept in filtered):
            filtered.append(c)
        if len(filtered) >= 8:
            break

    return filtered


def class1_quality(clip: np.ndarray) -> tuple[bool, str, ClipMetrics]:
    m = compute_metrics(clip)
    p95 = float(np.percentile(np.abs(clip), 95)) if len(clip) else 0.0
    prominence = m.peak / (m.median_abs + 1e-12)

    if m.peak < 0.01:
        return False, "peak_too_low", m
    if m.rms < 0.002:
        return False, "rms_too_low", m
    if m.crest_factor < 3.0:
        return False, "crest_too_low", m
    if prominence < 4.0:
        return False, "prominence_too_low", m
    if p95 < 0.002:
        return False, "too_quiet", m
    return True, "ok", m


def class0_quality(clip: np.ndarray) -> tuple[str, ClipMetrics]:
    m = compute_metrics(clip)
    prominence = m.peak / (m.median_abs + 1e-12)

    if m.peak < 0.003 or m.rms < 0.0005:
        return "silence", m

    # Very impulsive clips are suspicious for gunshot leakage and sent to noise.
    if m.crest_factor > 8.5 and m.attack_ratio > 12.0:
        return "uncertain_impulse", m
    if prominence > 15.0 and m.attack_ratio > 10.0:
        return "uncertain_impulse", m

    return "ok", m


def prepare_for_write(clip: np.ndarray) -> np.ndarray:
    clip = np.nan_to_num(clip.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    clip = clip - np.mean(clip)
    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
    if peak > 0.999:
        clip = clip / peak * 0.999
    return clip.astype(np.float32)


def write_wav(path: Path, clip: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), clip, sample_rate, subtype="PCM_16")
    if not path.exists() or path.stat().st_size <= 44:
        raise IOError(f"Empty/invalid file written: {path}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    start = args.project_root if args.project_root else Path.cwd()
    project_root = find_project_root(start)
    data_root = project_root / "Data"

    edge_root = data_root / "edge-collected-gunshot-audio"
    edge_nested = edge_root / "edge-collected-gunshot-audio"
    edge_dir = edge_nested if edge_nested.exists() else edge_root

    class1_dirs = [data_root / "gun", edge_dir]
    class0_dirs = [data_root / "sound"]
    if args.include_audio_folder:
        class0_dirs.append(data_root / "audio")

    target_samples = int(round(args.sample_rate * (args.target_ms / 1000.0)))
    pre_event_samples = int(round(args.sample_rate * 0.08))
    hop_samples = max(1, target_samples // 2)

    out_root = data_root / args.output_name
    class1_out = out_root / "class_1_gunshot"
    class0_out = out_root / "class_0_nongunshot"
    noise_c1_out = out_root / "noise" / "uncertain_from_class1"
    noise_c0_out = out_root / "noise" / "uncertain_from_class0"
    noise_silence_out = out_root / "noise" / "silence"
    noise_pitch_zero_out = out_root / "noise" / "pitch_zero"
    reports_out = out_root / "reports"

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for d in [
        class1_out,
        class0_out,
        noise_c1_out,
        noise_c0_out,
        noise_silence_out,
        noise_pitch_zero_out,
        reports_out,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    class1_sources = collect_wavs(class1_dirs, source_label="class1")
    class0_sources = collect_wavs(class0_dirs, source_label="class0")

    print("=" * 72)
    print("Strict 250 ms Dataset Builder")
    print("=" * 72)
    print(f"Project root          : {project_root}")
    print(f"Data root             : {data_root}")
    print(f"Output folder         : {out_root}")
    print(f"Target duration       : {args.target_ms} ms ({target_samples} samples)")
    print(f"Sample rate           : {args.sample_rate}")
    print(f"Class1 source files   : {len(class1_sources):,}")
    print(f"Class0 source files   : {len(class0_sources):,}")
    print("=" * 72)

    manifest_rows: list[dict] = []
    rejected_rows: list[dict] = []

    class1_written = 0
    noise_c1_written = 0
    class1_index = 0

    for rec in tqdm(class1_sources, desc="Class1 extraction", unit="file"):
        src_path: Path = rec["path"]
        src_stem = sanitize_name(src_path.stem)
        src_parent = sanitize_name(src_path.parent.name)
        source_key = f"{src_parent}_{src_stem}"

        try:
            y = load_audio(src_path, args.sample_rate)
        except Exception as exc:
            rejected_rows.append(
                {
                    "source_path": str(src_path),
                    "source_class": 1,
                    "reason": f"load_error: {exc}",
                }
            )
            continue

        if len(y) < 4:
            rejected_rows.append(
                {
                    "source_path": str(src_path),
                    "source_class": 1,
                    "reason": "too_short",
                }
            )
            continue

        candidates = get_class1_candidate_centers(y, args.sample_rate, target_samples)
        accepted_for_source = 0
        if not candidates:
            candidates = [int(np.argmax(np.abs(y)))]

        for cand_idx, center in enumerate(candidates):
            clip, src_start, src_end = get_fixed_window(
                y,
                center_sample=center,
                target_len=target_samples,
                pre_event_samples=pre_event_samples,
            )
            clip = prepare_for_write(clip)
            ok, reason, metrics = class1_quality(clip)

            if ok:
                class1_index += 1
                out_name = f"c1_{class1_index:07d}_{source_key}_k{cand_idx:02d}.wav"
                out_path = class1_out / out_name
                write_wav(out_path, clip, args.sample_rate)
                class1_written += 1
                accepted_for_source += 1

                manifest_rows.append(
                    {
                        "split": "class_1_gunshot",
                        "label": 1,
                        "output_path": str(out_path.relative_to(out_root)),
                        "source_path": str(src_path),
                        "source_dir": rec["source_dir"],
                        "candidate_index": cand_idx,
                        "start_sample": src_start,
                        "end_sample": src_end,
                        "samples": target_samples,
                        "duration_ms": args.target_ms,
                        "sample_rate": args.sample_rate,
                        "rms": metrics.rms,
                        "peak": metrics.peak,
                        "crest_factor": metrics.crest_factor,
                        "attack_ratio": metrics.attack_ratio,
                        "decision": "accepted",
                    }
                )

                if accepted_for_source >= args.max_class1_per_file:
                    break
            else:
                noise_name = f"noise_c1_{source_key}_k{cand_idx:02d}_{reason}.wav"
                noise_path = noise_c1_out / noise_name
                write_wav(noise_path, clip, args.sample_rate)
                noise_c1_written += 1

                manifest_rows.append(
                    {
                        "split": "noise_uncertain_from_class1",
                        "label": "noise",
                        "output_path": str(noise_path.relative_to(out_root)),
                        "source_path": str(src_path),
                        "source_dir": rec["source_dir"],
                        "candidate_index": cand_idx,
                        "start_sample": src_start,
                        "end_sample": src_end,
                        "samples": target_samples,
                        "duration_ms": args.target_ms,
                        "sample_rate": args.sample_rate,
                        "rms": metrics.rms,
                        "peak": metrics.peak,
                        "crest_factor": metrics.crest_factor,
                        "attack_ratio": metrics.attack_ratio,
                        "decision": reason,
                    }
                )

                # Keep only one uncertain clip per source candidate pass to avoid huge noise folder.
                if accepted_for_source == 0:
                    break

    class0_target = max(1, int(round(class1_written * args.class0_ratio)))
    class0_written = 0
    noise_c0_written = 0
    silence_written = 0
    class0_index = 0

    rng.shuffle(class0_sources)

    for rec in tqdm(class0_sources, desc="Class0 extraction", unit="file"):
        if class0_written >= class0_target:
            break

        src_path: Path = rec["path"]
        src_stem = sanitize_name(src_path.stem)
        src_parent = sanitize_name(src_path.parent.name)
        source_key = f"{src_parent}_{src_stem}"

        try:
            y = load_audio(src_path, args.sample_rate)
        except Exception as exc:
            rejected_rows.append(
                {
                    "source_path": str(src_path),
                    "source_class": 0,
                    "reason": f"load_error: {exc}",
                }
            )
            continue

        if len(y) < target_samples:
            clip = librosa.util.fix_length(y, size=target_samples)
            starts = np.array([0], dtype=np.int64)
        else:
            starts = np.arange(0, len(y) - target_samples + 1, hop_samples, dtype=np.int64)
            rng.shuffle(starts)
            # Keep extraction balanced across files.
            starts = starts[: min(len(starts), 600)]

        kept_from_file = 0
        for win_idx, start_idx in enumerate(starts):
            if class0_written >= class0_target:
                break

            if len(y) < target_samples:
                clip = librosa.util.fix_length(y, size=target_samples)
                src_start = 0
                src_end = len(y)
            else:
                src_start = int(start_idx)
                src_end = int(start_idx + target_samples)
                clip = y[src_start:src_end]

            clip = prepare_for_write(clip)
            verdict, metrics = class0_quality(clip)

            if verdict == "ok":
                class0_index += 1
                out_name = f"c0_{class0_index:07d}_{source_key}_w{win_idx:03d}.wav"
                out_path = class0_out / out_name
                write_wav(out_path, clip, args.sample_rate)
                class0_written += 1
                kept_from_file += 1

                manifest_rows.append(
                    {
                        "split": "class_0_nongunshot",
                        "label": 0,
                        "output_path": str(out_path.relative_to(out_root)),
                        "source_path": str(src_path),
                        "source_dir": rec["source_dir"],
                        "candidate_index": win_idx,
                        "start_sample": src_start,
                        "end_sample": src_end,
                        "samples": target_samples,
                        "duration_ms": args.target_ms,
                        "sample_rate": args.sample_rate,
                        "rms": metrics.rms,
                        "peak": metrics.peak,
                        "crest_factor": metrics.crest_factor,
                        "attack_ratio": metrics.attack_ratio,
                        "decision": "accepted",
                    }
                )
            elif verdict == "uncertain_impulse":
                noise_name = f"noise_c0_{source_key}_w{win_idx:03d}_impulse.wav"
                noise_path = noise_c0_out / noise_name
                write_wav(noise_path, clip, args.sample_rate)
                noise_c0_written += 1

                manifest_rows.append(
                    {
                        "split": "noise_uncertain_from_class0",
                        "label": "noise",
                        "output_path": str(noise_path.relative_to(out_root)),
                        "source_path": str(src_path),
                        "source_dir": rec["source_dir"],
                        "candidate_index": win_idx,
                        "start_sample": src_start,
                        "end_sample": src_end,
                        "samples": target_samples,
                        "duration_ms": args.target_ms,
                        "sample_rate": args.sample_rate,
                        "rms": metrics.rms,
                        "peak": metrics.peak,
                        "crest_factor": metrics.crest_factor,
                        "attack_ratio": metrics.attack_ratio,
                        "decision": verdict,
                    }
                )
            else:
                silence_name = f"silence_{source_key}_w{win_idx:03d}.wav"
                silence_path = noise_silence_out / silence_name
                write_wav(silence_path, clip, args.sample_rate)
                silence_written += 1

                manifest_rows.append(
                    {
                        "split": "noise_silence",
                        "label": "noise",
                        "output_path": str(silence_path.relative_to(out_root)),
                        "source_path": str(src_path),
                        "source_dir": rec["source_dir"],
                        "candidate_index": win_idx,
                        "start_sample": src_start,
                        "end_sample": src_end,
                        "samples": target_samples,
                        "duration_ms": args.target_ms,
                        "sample_rate": args.sample_rate,
                        "rms": metrics.rms,
                        "peak": metrics.peak,
                        "crest_factor": metrics.crest_factor,
                        "attack_ratio": metrics.attack_ratio,
                        "decision": verdict,
                    }
                )

            if kept_from_file >= 80:
                break

    # Create pitch-zero (perfect silence) noise clips.
    for i in range(args.zero_noise_count):
        clip = np.zeros(target_samples, dtype=np.float32)
        out_name = f"pitch_zero_{i + 1:06d}.wav"
        out_path = noise_pitch_zero_out / out_name
        write_wav(out_path, clip, args.sample_rate)
        manifest_rows.append(
            {
                "split": "noise_pitch_zero",
                "label": "noise",
                "output_path": str(out_path.relative_to(out_root)),
                "source_path": "synthetic::pitch_zero",
                "source_dir": "generated",
                "candidate_index": i,
                "start_sample": 0,
                "end_sample": target_samples,
                "samples": target_samples,
                "duration_ms": args.target_ms,
                "sample_rate": args.sample_rate,
                "rms": 0.0,
                "peak": 0.0,
                "crest_factor": 0.0,
                "attack_ratio": 0.0,
                "decision": "generated_pitch_zero",
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    manifest_path = reports_out / "manifest.csv"
    rejected_path = reports_out / "rejected_sources.csv"
    summary_path = reports_out / "summary.json"

    manifest_df.to_csv(manifest_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)

    summary = {
        "project_root": str(project_root),
        "data_root": str(data_root),
        "output_root": str(out_root),
        "params": {
            "sample_rate": args.sample_rate,
            "target_ms": args.target_ms,
            "target_samples": target_samples,
            "class0_ratio": args.class0_ratio,
            "max_class1_per_file": args.max_class1_per_file,
            "include_audio_folder": args.include_audio_folder,
            "zero_noise_count": args.zero_noise_count,
            "seed": args.seed,
        },
        "counts": {
            "source_files_class1": len(class1_sources),
            "source_files_class0": len(class0_sources),
            "class1_written": class1_written,
            "class0_written": class0_written,
            "noise_uncertain_from_class1": noise_c1_written,
            "noise_uncertain_from_class0": noise_c0_written,
            "noise_silence": silence_written,
            "noise_pitch_zero": args.zero_noise_count,
            "rejected_sources": len(rejected_rows),
            "manifest_rows": len(manifest_rows),
        },
        "files": {
            "manifest_csv": str(manifest_path),
            "rejected_sources_csv": str(rejected_path),
            "summary_json": str(summary_path),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Build completed")
    print("=" * 72)
    print(f"Class 1 clips            : {class1_written:,}")
    print(f"Class 0 clips            : {class0_written:,}")
    print(f"Noise (from class1)      : {noise_c1_written:,}")
    print(f"Noise (from class0)      : {noise_c0_written:,}")
    print(f"Noise (silence)          : {silence_written:,}")
    print(f"Noise (pitch zero)       : {args.zero_noise_count:,}")
    print(f"Rejected source files    : {len(rejected_rows):,}")
    print(f"Manifest rows            : {len(manifest_rows):,}")
    print("=" * 72)
    print(f"Manifest: {manifest_path}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
