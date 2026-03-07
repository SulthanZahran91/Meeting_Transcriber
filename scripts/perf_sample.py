from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
import time

import av
import psutil
from rich.console import Console

from meeting_transcriber.config import TranscriberConfig
from meeting_transcriber.transcribe import (
    Segment,
    _load_model,
    clear_model_cache,
    transcribe,
)
from meeting_transcriber.translate import (
    _load_translation_model,
    clear_translation_cache,
    translate_segments,
)


def _silent_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


def _audio_duration_seconds(path: Path) -> float:
    with av.open(str(path)) as container:
        if container.duration is not None:
            return float(container.duration / 1_000_000.0)
    return 0.0


def main() -> None:
    sample_path = Path("samples/korean_sample.ogg")
    if not sample_path.exists():
        raise SystemExit(f"Missing sample file: {sample_path}")

    console = _silent_console()
    cfg = TranscriberConfig(
        whisper_model="tiny",
        device="cpu",
        compute_type="int8",
        beam_size=1,
        vad_filter=True,
    )

    # Model load times
    clear_model_cache()
    started = time.perf_counter()
    _load_model(cfg, console=console)
    asr_load_s = time.perf_counter() - started

    clear_translation_cache()
    started = time.perf_counter()
    _load_translation_model(cfg, console=console)
    mt_load_s = time.perf_counter() - started

    # Transcription throughput on sample
    audio_seconds = _audio_duration_seconds(sample_path)
    started = time.perf_counter()
    segments = transcribe(sample_path, cfg, console=console)
    transcribe_s = time.perf_counter() - started
    transcribe_x = (audio_seconds / transcribe_s) if transcribe_s > 0 else 0.0

    # Translation throughput by batch size
    base_text = segments[0].text if segments else "대한민국"
    many_segments = [
        Segment(float(i), float(i + 1), base_text) for i in range(160)
    ]
    translation_perf: dict[str, dict[str, float]] = {}
    for batch_size in (8, 12, 16):
        started = time.perf_counter()
        translate_segments(
            many_segments, cfg, console=console, batch_size=batch_size
        )
        elapsed = time.perf_counter() - started
        translation_perf[str(batch_size)] = {
            "seconds": elapsed,
            "segments_per_second": (len(many_segments) / elapsed) if elapsed > 0 else 0.0,
        }

    # VAD impact
    no_vad_cfg = replace(cfg, vad_filter=False)
    started = time.perf_counter()
    transcribe(sample_path, no_vad_cfg, console=console)
    transcribe_no_vad_s = time.perf_counter() - started
    vad_delta_pct = (
        ((transcribe_no_vad_s - transcribe_s) / transcribe_no_vad_s) * 100.0
        if transcribe_no_vad_s > 0
        else 0.0
    )

    # Cache/reuse check
    model_a = _load_model(cfg, console=console)
    model_b = _load_model(cfg, console=console)
    model_reuse = model_a is model_b

    # Memory growth check across repeated runs
    process = psutil.Process()
    rss_before = process.memory_info().rss
    for _ in range(20):
        transcribe(sample_path, cfg, console=console)
    rss_after = process.memory_info().rss
    rss_delta_mb = (rss_after - rss_before) / (1024 * 1024)

    report = {
        "sample_file": str(sample_path),
        "audio_duration_seconds": audio_seconds,
        "asr_model_load_seconds": asr_load_s,
        "mt_model_load_seconds": mt_load_s,
        "transcription_seconds": transcribe_s,
        "transcription_realtime_factor": transcribe_x,
        "translation_perf": translation_perf,
        "transcribe_vad_true_seconds": transcribe_s,
        "transcribe_vad_false_seconds": transcribe_no_vad_s,
        "vad_time_saved_percent": vad_delta_pct,
        "model_cache_reuse": model_reuse,
        "rss_delta_mb_over_20_runs": rss_delta_mb,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

