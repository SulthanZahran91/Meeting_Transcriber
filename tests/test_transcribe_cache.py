from pathlib import Path
import types

from rich.console import Console

from meeting_transcriber.config import TranscriberConfig
from meeting_transcriber.transcribe import _MODEL_CACHE, _load_model


def test_whisper_model_load_cached(monkeypatch, tmp_path: Path) -> None:
    calls = {"init": 0}

    class FakeWhisperModel:
        def __init__(self, **_: object) -> None:
            calls["init"] += 1

        def transcribe(self, *_: object, **__: object) -> tuple[list[object], object]:
            return [], None

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", fake_module)
    _MODEL_CACHE.clear()

    cfg = TranscriberConfig(whisper_model="small", device="cpu", compute_type="int8")
    console = Console(record=True)
    m1 = _load_model(cfg, console=console)
    m2 = _load_model(cfg, console=console)

    assert m1 is m2
    assert calls["init"] == 1
