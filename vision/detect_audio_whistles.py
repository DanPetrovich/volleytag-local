#!/usr/bin/env python3
"""Find whistle-like tonal peaks in a video's audio track."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, stft


def robust_z(values: np.ndarray) -> np.ndarray:
    center = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - center))
    return (values - center) / max(float(scale), 1e-6)


def read_audio(video: Path, sample_rate: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav") as output:
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
            "-ar", str(sample_rate), "-c:a", "pcm_s16le", output.name,
        ], check=True)
        rate, audio = wavfile.read(output.name)
    if rate != sample_rate:
        raise ValueError(f"unexpected sample rate: {rate}")
    return audio.astype(np.float32) / max(float(np.iinfo(audio.dtype).max), 1.0)


def candidates(audio: np.ndarray, sample_rate: int, max_per_minute: float) -> list[dict]:
    frequencies, times, spectrum = stft(
        audio, fs=sample_rate, window="hann", nperseg=1024, noverlap=768,
        boundary=None, padded=False,
    )
    power = np.abs(spectrum) ** 2 + 1e-12
    whistle = (frequencies >= 2200) & (frequencies <= 5200)
    audible = (frequencies >= 500) & (frequencies <= 7500)
    band = power[whistle]
    band_energy = band.sum(axis=0)
    audible_energy = power[audible].sum(axis=0)
    concentration = band.max(axis=0) / np.maximum(band.mean(axis=0), 1e-12)
    share = band_energy / np.maximum(audible_energy, 1e-12)
    dominant = frequencies[whistle][np.argmax(band, axis=0)]

    # A whistle is energetic, narrow-band and concentrated above speech fundamentals.
    raw_score = (
        robust_z(np.log10(band_energy))
        + 1.6 * robust_z(np.log1p(concentration))
        + 1.2 * robust_z(share)
    )
    score = median_filter(raw_score, size=3, mode="nearest")
    frame_rate = 1.0 / np.median(np.diff(times))
    indexes, _ = find_peaks(score, distance=max(1, int(frame_rate * .45)), prominence=.35)
    duration_minutes = len(audio) / sample_rate / 60.0
    keep = max(1, round(duration_minutes * max_per_minute))
    indexes = indexes[np.argsort(score[indexes])[-keep:]]
    indexes = indexes[np.argsort(times[indexes])]
    return [{
        "time_s": round(float(times[index]), 3),
        "score": round(float(score[index]), 4),
        "dominant_hz": round(float(dominant[index]), 1),
        "band_share": round(float(share[index]), 4),
        "concentration": round(float(concentration[index]), 3),
    } for index in indexes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-per-minute", type=float, default=6.0)
    args = parser.parse_args()
    audio = read_audio(args.video, args.sample_rate)
    rows = candidates(audio, args.sample_rate, args.max_per_minute)
    payload = {
        "schema_version": 1,
        "kind": "volleytag.audio-whistle-candidates",
        "video": str(args.video),
        "sample_rate": args.sample_rate,
        "duration_s": round(len(audio) / args.sample_rate, 3),
        "max_per_minute": args.max_per_minute,
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "duration_s": payload["duration_s"], "candidates": len(rows),
        "candidates_per_minute": round(len(rows) / payload["duration_s"] * 60, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
