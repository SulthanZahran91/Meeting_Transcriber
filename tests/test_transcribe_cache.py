from pathlib import Path
import sys
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


def test_whisper_model_refreshes_stale_snapshot_cache(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    snapshot_calls: list[dict[str, object]] = []

    class FakeWhisperModel:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError(
                    "An error happened while trying to locate the files on the Hub and "
                    "we cannot find the appropriate snapshot folder for the specified "
                    "revision on the local disk."
                )

        def transcribe(self, *_: object, **__: object) -> tuple[list[object], object]:
            return [], None

    fake_whisper_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    fake_hf_hub = types.SimpleNamespace(
        snapshot_download=lambda **kwargs: snapshot_calls.append(kwargs) or "/tmp/whisper-model"
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_whisper_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    monkeypatch.setattr(
        "meeting_transcriber.transcribe._whisper_model_dir",
        lambda _model_name: tmp_path / "tiny",
    )
    _MODEL_CACHE.clear()

    cfg = TranscriberConfig(whisper_model="tiny", device="cpu", compute_type="int8")
    _load_model(cfg, console=Console(record=True))

    assert calls[0]["model_size_or_path"] == "tiny"
    assert calls[1]["model_size_or_path"] == str(tmp_path / "tiny")
    assert snapshot_calls == [
        {
            "repo_id": "Systran/faster-whisper-tiny",
            "local_dir": str(tmp_path / "tiny"),
            "force_download": True,
            "local_files_only": False,
        }
    ]
