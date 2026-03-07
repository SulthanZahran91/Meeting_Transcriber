from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from rich.console import Console

from meeting_transcriber.config import TranscriberConfig
from meeting_transcriber.transcribe import transcribe


@dataclass
class _FakeWhisperSegment:
    start: float
    end: float
    text: str


class _FakeWhisperModel:
    def __init__(self, segments: list[_FakeWhisperSegment], fail: Exception | None = None) -> None:
        self._segments = segments
        self._fail = fail

    def transcribe(self, *_: object, **__: object) -> tuple[list[_FakeWhisperSegment], object]:
        if self._fail is not None:
            raise self._fail
        return self._segments, None


def test_transcribe_logs_every_50_segments(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    segments = [
        _FakeWhisperSegment(float(i), float(i + 1), f"text {i}") for i in range(100)
    ]
    monkeypatch.setattr(
        "meeting_transcriber.transcribe._load_model",
        lambda *_args, **_kwargs: _FakeWhisperModel(segments),
    )
    console = Console(record=True)

    out = transcribe(
        audio,
        TranscriberConfig(whisper_model="tiny", device="cpu", compute_type="int8", beam_size=1),
        console=console,
    )

    rendered = console.export_text()
    assert len(out) == 100
    assert "Transcribed 50 segments" in rendered
    assert "Transcribed 100 segments" in rendered


def test_transcribe_oom_message_is_humanized(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setattr(
        "meeting_transcriber.transcribe._load_model",
        lambda *_args, **_kwargs: _FakeWhisperModel([], fail=RuntimeError("out of memory")),
    )

    with pytest.raises(RuntimeError, match="Try --model small or --model tiny"):
        transcribe(
            audio,
            TranscriberConfig(
                whisper_model="tiny",
                device="cuda",
                compute_type="float16",
                beam_size=1,
            ),
        )

