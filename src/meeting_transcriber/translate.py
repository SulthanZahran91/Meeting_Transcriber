from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from meeting_transcriber.config import TranscriberConfig
from meeting_transcriber.transcribe import Segment


@dataclass(frozen=True)
class TranslatedSegment:
    start: float
    end: float
    korean: str
    english: str


_TRANSLATION_CACHE: dict[str, tuple[object, object]] = {}
DEFAULT_BATCH_SIZE = 12


def translate_segments(
    segments: list[Segment],
    config: TranscriberConfig,
    console: Optional[Console] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[TranslatedSegment]:
    console = console or Console()
    if not segments:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    tokenizer, model = _load_translation_model(config, console=console)

    translated: list[Optional[TranslatedSegment]] = [None] * len(segments)
    active_indices: list[int] = []
    active_texts: list[str] = []
    for idx, seg in enumerate(segments):
        text = seg.text.strip()
        if text:
            active_indices.append(idx)
            active_texts.append(text)
        else:
            translated[idx] = TranslatedSegment(seg.start, seg.end, seg.text, "")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Translating", total=len(segments))
        completed_non_empty = 0
        for idx in range(0, len(active_texts), batch_size):
            chunk_text = active_texts[idx : idx + batch_size]
            chunk_indexes = active_indices[idx : idx + batch_size]
            decoded = _run_translation_batch(tokenizer, model, chunk_text)
            for original_idx, english in zip(chunk_indexes, decoded, strict=True):
                seg = segments[original_idx]
                translated[original_idx] = TranslatedSegment(
                    start=seg.start,
                    end=seg.end,
                    korean=seg.text,
                    english=english.strip(),
                )
            completed_non_empty += len(chunk_indexes)
            progress.advance(task, len(chunk_indexes))

        # Empty segments were already assigned and still count toward total progress.
        progress.advance(task, len(segments) - completed_non_empty)

    return [item for item in translated if item is not None]


def _load_translation_model(
    config: TranscriberConfig, console: Console
) -> tuple[object, object]:
    cached = _TRANSLATION_CACHE.get(config.translation_model)
    if cached is not None:
        return cached

    try:
        from transformers import MarianMTModel, MarianTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Failed to import transformers. Install dependencies with `uv sync`."
        ) from exc

    with console.status(f"Loading translation model '{config.translation_model}'..."):
        tokenizer = MarianTokenizer.from_pretrained(config.translation_model)
        try:
            model = MarianMTModel.from_pretrained(config.translation_model)
        except ImportError as exc:
            raise RuntimeError(
                "Translation backend requires PyTorch. Install it with "
                "`uv sync --extra gpu` and retry."
            ) from exc
        model.eval()

    _TRANSLATION_CACHE[config.translation_model] = (tokenizer, model)
    return tokenizer, model


def clear_translation_cache() -> None:
    _TRANSLATION_CACHE.clear()


def _run_translation_batch(tokenizer: object, model: object, inputs: list[str]) -> list[str]:
    if not inputs:
        return []
    encoded = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    try:
        import torch

        context = torch.inference_mode()
    except Exception:
        context = nullcontext()

    with context:
        generated = model.generate(**encoded)

    return tokenizer.batch_decode(generated, skip_special_tokens=True)
