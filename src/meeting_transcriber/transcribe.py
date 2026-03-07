from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

from meeting_transcriber.config import TranscriberConfig


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


def transcribe(
    audio_path: Path, config: TranscriberConfig, console: Optional[Console] = None
) -> list[Segment]:
    console = console or Console()
    model = _load_model(config, console=console)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        segments_iter, _ = model.transcribe(
            str(audio_path),
            language=config.language,
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
            vad_parameters={"min_silence_duration_ms": 500},
        )
    except Exception as exc:
        raise RuntimeError(_humanize_asr_error(exc)) from exc

    collected: list[Segment] = []
    for idx, part in enumerate(segments_iter, start=1):
        text = (part.text or "").strip()
        if text:
            collected.append(Segment(start=float(part.start), end=float(part.end), text=text))
        if idx % 50 == 0:
            console.print(f"[cyan]Transcribed {idx} segments...[/cyan]")
    return collected


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def _load_model(config: TranscriberConfig, console: Console) -> object:
    key = (config.whisper_model, config.device, config.compute_type)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "Failed to import faster-whisper. Install dependencies with `uv sync`."
        ) from exc

    with console.status(
        f"Loading Whisper model '{config.whisper_model}' on {config.device}..."
    ):
        model = WhisperModel(
            model_size_or_path=config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
        )
    _MODEL_CACHE[key] = model
    return model


def _humanize_asr_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "out of memory" in lowered:
        return (
            f"ASR out of memory: {message}. "
            "Try --model small or --model tiny. "
            "If using CUDA, retry with --device cpu."
        )
    if "download" in lowered or "network" in lowered:
        return (
            f"ASR model download failed: {message}. "
            "Retry once network is available or preload model cache."
        )
    return f"ASR failed: {message}"

