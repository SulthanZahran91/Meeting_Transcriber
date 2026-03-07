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
        try:
            model = _load_whisper_model_once(
                whisper_model_cls=WhisperModel,
                model_name=config.whisper_model,
                device=config.device,
                compute_type=config.compute_type,
                console=console,
            )
        except Exception as exc:
            raise RuntimeError(_humanize_asr_error(exc)) from exc
    _MODEL_CACHE[key] = model
    return model


def _load_whisper_model_once(
    whisper_model_cls: object,
    model_name: str,
    device: str,
    compute_type: str,
    console: Console,
) -> object:
    try:
        return whisper_model_cls(
            model_size_or_path=model_name,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        if not _is_hf_snapshot_cache_error(exc):
            raise

        console.print(
            "[yellow]Detected stale ASR model cache. "
            "Refreshing Hugging Face snapshot and retrying once...[/yellow]"
        )
        snapshot_path = _refresh_whisper_snapshot(model_name=model_name)
        if snapshot_path is None:
            raise

        return whisper_model_cls(
            model_size_or_path=snapshot_path,
            device=device,
            compute_type=compute_type,
        )


def _refresh_whisper_snapshot(model_name: str) -> str | None:
    repo_id = _resolve_whisper_repo_id(model_name)
    if repo_id is None:
        return None

    model_dir = _whisper_model_dir(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        force_download=True,
        local_files_only=False,
    )
    return str(model_dir)


def _whisper_model_dir(model_name: str) -> Path:
    safe_name = model_name.replace("/", "--")
    return Path.home() / ".cache" / "meeting-transcriber" / "models" / safe_name


def _resolve_whisper_repo_id(model_name: str) -> str | None:
    supported = {"tiny", "small", "medium", "large-v3"}
    if model_name not in supported:
        return None
    return f"Systran/faster-whisper-{model_name}"


def _is_hf_snapshot_cache_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return (
        "snapshot folder" in lowered
        or (
            "locate the files on the hub" in lowered
            and "local disk" in lowered
            and "revision" in lowered
        )
    )


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
    if _is_hf_snapshot_cache_error(exc):
        return (
            f"ASR model cache is incomplete/corrupted: {message}. "
            "Retry once with internet access to refresh cache. "
            "If it persists, clear local Hugging Face cache and retry."
        )
    return f"ASR failed: {message}"
