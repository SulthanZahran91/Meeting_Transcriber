from __future__ import annotations

from pathlib import Path
import sys
import types

from meeting_transcriber.config import TranscriberConfig
from meeting_transcriber.transcribe import Segment
from meeting_transcriber import translate
from meeting_transcriber.translate import translate_segments
from rich.console import Console


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


def test_load_translation_model_retries_snapshot_cache_error(monkeypatch) -> None:
    snapshot_calls: list[dict[str, object]] = []

    class _FakeTokenizer:
        calls = 0

        @classmethod
        def from_pretrained(cls, *_args: object, **kwargs: object) -> object:
            cls.calls += 1
            if cls.calls == 1:
                raise RuntimeError(
                    "An error happened while trying to locate the files on the Hub and "
                    "we cannot find the appropriate snapshot folder for the specified "
                    "revision on the local disk."
                )
            assert kwargs.get("force_download") is True
            return object()

    class _FakeLoadedModel:
        def eval(self) -> None:
            return None

    class _FakeModel:
        calls = 0

        @classmethod
        def from_pretrained(cls, *_args: object, **kwargs: object) -> _FakeLoadedModel:
            cls.calls += 1
            assert kwargs.get("force_download") is True
            return _FakeLoadedModel()

    fake_transformers = types.SimpleNamespace(
        MarianTokenizer=_FakeTokenizer,
        MarianMTModel=_FakeModel,
    )
    fake_hf_hub = types.SimpleNamespace(
        snapshot_download=lambda **kwargs: snapshot_calls.append(kwargs)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    monkeypatch.setattr(
        "meeting_transcriber.translate._translation_model_dir",
        lambda _model_name: Path("/tmp/translation-model"),
    )
    translate.clear_translation_cache()

    translate._load_translation_model(  # type: ignore[attr-defined]
        TranscriberConfig(translation_model="demo/model"),
        console=Console(record=True),
    )

    assert _FakeTokenizer.calls == 2
    assert _FakeModel.calls == 1
    assert snapshot_calls == [
        {
            "repo_id": "demo/model",
            "local_dir": "/tmp/translation-model",
            "force_download": True,
            "local_files_only": False,
        }
    ]
