from __future__ import annotations

from pathlib import Path
import shutil
import wave

import pytest

from meeting_transcriber.extract import extract_audio


def test_extract_audio_missing_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"dummy")
    monkeypatch.setattr("meeting_transcriber.extract.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
        extract_audio(source)


def test_extract_audio_temp_output_created(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"dummy")
    monkeypatch.setattr("meeting_transcriber.extract.shutil.which", lambda _: "/usr/bin/ffmpeg")

    def fake_input(path: str) -> dict[str, str]:
        return {"in": path}

    def fake_output(stream: dict[str, str], out: str, **_: object) -> dict[str, str]:
        stream["out"] = out
        return stream

    def fake_run(stream: dict[str, str], **_: object) -> None:
        Path(stream["out"]).write_bytes(b"RIFF")

    monkeypatch.setattr("meeting_transcriber.extract.ffmpeg.input", fake_input)
    monkeypatch.setattr("meeting_transcriber.extract.ffmpeg.output", fake_output)
    monkeypatch.setattr("meeting_transcriber.extract.ffmpeg.run", fake_run)

    out_path = extract_audio(source)

    assert out_path.exists()
    assert out_path.suffix == ".wav"
    out_path.unlink(missing_ok=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_audio_integration_real_ffmpeg(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    out = extract_audio(source, output_path=tmp_path / "out.wav")
    assert out.exists()
    assert out.stat().st_size > 0

