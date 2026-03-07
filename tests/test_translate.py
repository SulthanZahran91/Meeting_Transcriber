from __future__ import annotations

from meeting_transcriber.config import TranscriberConfig
from meeting_transcriber.transcribe import Segment
from meeting_transcriber.translate import translate_segments


def test_translate_batches_and_keeps_alignment(monkeypatch) -> None:
    batch_calls: list[list[str]] = []
    monkeypatch.setattr(
        "meeting_transcriber.translate._load_translation_model",
        lambda *_args, **_kwargs: (object(), object()),
    )

    def fake_run(_tokenizer: object, _model: object, inputs: list[str]) -> list[str]:
        batch_calls.append(inputs[:])
        return [f"EN:{txt}" for txt in inputs]

    monkeypatch.setattr("meeting_transcriber.translate._run_translation_batch", fake_run)

    segments = [
        Segment(0.0, 1.0, "하나"),
        Segment(1.0, 2.0, ""),
        Segment(2.0, 3.0, "둘"),
        Segment(3.0, 4.0, "셋"),
        Segment(4.0, 5.0, "넷"),
    ]
    out = translate_segments(
        segments,
        TranscriberConfig(),
        batch_size=2,
    )

    assert len(out) == 5
    assert batch_calls == [["하나", "둘"], ["셋", "넷"]]
    assert out[1].english == ""
    assert out[2].english == "EN:둘"
    assert out[3].start == 3.0
    assert out[3].end == 4.0


def test_translate_skips_whitespace_segments(monkeypatch) -> None:
    monkeypatch.setattr(
        "meeting_transcriber.translate._load_translation_model",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        "meeting_transcriber.translate._run_translation_batch",
        lambda _tokenizer, _model, inputs: [f"EN:{txt}" for txt in inputs],
    )

    segments = [
        Segment(0.0, 1.0, "  "),
        Segment(1.0, 2.0, "\n"),
        Segment(2.0, 3.0, "정상"),
    ]
    out = translate_segments(segments, TranscriberConfig(), batch_size=8)

    assert [item.english for item in out] == ["", "", "EN:정상"]

