from pathlib import Path

from meeting_transcriber.config import (
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_TRANSLATION_CONTEXT_WINDOW,
    DEFAULT_TRANSLATION_MODEL,
    TranscriberConfig,
    resolve_auto_config,
)
from meeting_transcriber.hardware import ModelRecommendation


def test_resolve_auto_config_uses_recommendation_for_auto_fields() -> None:
    cfg = TranscriberConfig(
        whisper_model="auto",
        device="auto",
        compute_type="auto",
        output_dir=Path("out"),
    )
    rec = ModelRecommendation(
        whisper_model="medium",
        device="cpu",
        compute_type="int8",
        beam_size=3,
        reason="test",
    )

    resolved = resolve_auto_config(cfg, rec)

    assert resolved.whisper_model == "medium"
    assert resolved.device == "cpu"
    assert resolved.compute_type == "int8"
    assert resolved.beam_size == 3
    assert resolved.output_dir == Path("out")


def test_resolve_auto_config_preserves_manual_values() -> None:
    cfg = TranscriberConfig(
        whisper_model="small",
        device="cuda",
        compute_type="float16",
        beam_size=7,
    )
    rec = ModelRecommendation(
        whisper_model="tiny",
        device="cpu",
        compute_type="int8",
        beam_size=1,
        reason="test",
    )

    resolved = resolve_auto_config(cfg, rec)

    assert resolved.whisper_model == "small"
    assert resolved.device == "cuda"
    assert resolved.compute_type == "float16"
    assert resolved.beam_size == 7


def test_transcriber_config_defaults_match_accuracy_first_translation() -> None:
    cfg = TranscriberConfig()

    assert cfg.translation_model == DEFAULT_TRANSLATION_MODEL
    assert cfg.output_format == DEFAULT_OUTPUT_FORMAT
    assert cfg.translation_context_window == DEFAULT_TRANSLATION_CONTEXT_WINDOW
