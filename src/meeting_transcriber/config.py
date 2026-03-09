from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from meeting_transcriber.hardware import ModelRecommendation

DEFAULT_TRANSLATION_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_OUTPUT_FORMAT = "srt"
DEFAULT_TRANSLATION_CONTEXT_WINDOW = 3
DEFAULT_TRANSLATION_MAX_NEW_TOKENS = 256


@dataclass(frozen=True)
class TranscriberConfig:
    whisper_model: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    language: str = "ko"
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    output_format: str = DEFAULT_OUTPUT_FORMAT
    output_dir: Path = Path("output")
    glossary_path: Path = Path("glossary.json")
    vad_filter: bool = True
    max_segment_length: int = 500
    translation_context_window: int = DEFAULT_TRANSLATION_CONTEXT_WINDOW
    translation_max_new_tokens: int = DEFAULT_TRANSLATION_MAX_NEW_TOKENS


def resolve_auto_config(
    config: TranscriberConfig, recommendation: ModelRecommendation
) -> TranscriberConfig:
    whisper_model = (
        recommendation.whisper_model
        if config.whisper_model == "auto"
        else config.whisper_model
    )
    device = recommendation.device if config.device == "auto" else config.device
    compute_type = (
        recommendation.compute_type
        if config.compute_type == "auto"
        else config.compute_type
    )
    beam_size = (
        recommendation.beam_size if config.whisper_model == "auto" else config.beam_size
    )
    return TranscriberConfig(
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        language=config.language,
        translation_model=config.translation_model,
        output_format=config.output_format,
        output_dir=config.output_dir,
        glossary_path=config.glossary_path,
        vad_filter=config.vad_filter,
        max_segment_length=config.max_segment_length,
        translation_context_window=config.translation_context_window,
        translation_max_new_tokens=config.translation_max_new_tokens,
    )
