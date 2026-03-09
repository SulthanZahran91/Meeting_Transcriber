from pathlib import Path

import pytest
from rich.console import Console
import typer
from typer.testing import CliRunner

from meeting_transcriber import cli
from meeting_transcriber.config import TranscriberConfig
from meeting_transcriber.hardware import HardwareProfile, ModelRecommendation


runner = CliRunner()


def test_show_hardware_exits_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "detect_hardware",
        lambda: HardwareProfile(
            device="cpu",
            gpu_name=None,
            gpu_vram_gb=None,
            ram_gb=16.0,
            cpu_name="cpu",
            cpu_cores=8,
        ),
    )
    monkeypatch.setattr(
        cli,
        "recommend_config",
        lambda *args, **kwargs: ModelRecommendation(
            whisper_model="medium",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            reason="ok",
        ),
    )
    monkeypatch.setattr(cli, "print_hardware_summary", lambda *args, **kwargs: None)

    result = runner.invoke(cli.app, ["transcribe", "--show-hardware"])

    assert result.exit_code == 0


def test_single_file_invokes_processing_once(monkeypatch, tmp_path: Path) -> None:
    in_file = tmp_path / "meeting.wav"
    in_file.write_bytes(b"wav")
    calls = {"count": 0}

    monkeypatch.setattr(
        cli,
        "detect_hardware",
        lambda: HardwareProfile(
            device="cpu",
            gpu_name=None,
            gpu_vram_gb=None,
            ram_gb=16.0,
            cpu_name="cpu",
            cpu_cores=8,
        ),
    )
    monkeypatch.setattr(
        cli,
        "recommend_config",
        lambda *args, **kwargs: ModelRecommendation(
            whisper_model="medium",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            reason="ok",
        ),
    )
    monkeypatch.setattr(cli, "print_hardware_summary", lambda *args, **kwargs: None)

    def fake_process(input_path: Path, config: TranscriberConfig, no_translate: bool) -> Path:
        calls["count"] += 1
        assert input_path == in_file
        assert config.whisper_model == "medium"
        return tmp_path / "meeting.html"

    monkeypatch.setattr(cli, "_process_one_file", fake_process)
    result = runner.invoke(cli.app, ["transcribe", str(in_file)])

    assert result.exit_code == 0
    assert calls["count"] == 1


def test_batch_empty_directory_is_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "detect_hardware",
        lambda: HardwareProfile(
            device="cpu",
            gpu_name=None,
            gpu_vram_gb=None,
            ram_gb=16.0,
            cpu_name="cpu",
            cpu_cores=8,
        ),
    )
    monkeypatch.setattr(
        cli,
        "recommend_config",
        lambda *args, **kwargs: ModelRecommendation(
            whisper_model="medium",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            reason="ok",
        ),
    )
    monkeypatch.setattr(cli, "print_hardware_summary", lambda *args, **kwargs: None)

    result = runner.invoke(cli.app, ["transcribe", "--batch", str(tmp_path)])

    assert result.exit_code == 1


def test_batch_directory_processes_all_files(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"a")
    (tmp_path / "b.mp3").write_bytes(b"b")
    calls: list[Path] = []

    monkeypatch.setattr(
        cli,
        "detect_hardware",
        lambda: HardwareProfile(
            device="cpu",
            gpu_name=None,
            gpu_vram_gb=None,
            ram_gb=16.0,
            cpu_name="cpu",
            cpu_cores=8,
        ),
    )
    monkeypatch.setattr(
        cli,
        "recommend_config",
        lambda *args, **kwargs: ModelRecommendation(
            whisper_model="medium",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            reason="ok",
        ),
    )
    monkeypatch.setattr(cli, "print_hardware_summary", lambda *args, **kwargs: None)

    def fake_process(input_path: Path, config: TranscriberConfig, no_translate: bool) -> Path:
        calls.append(input_path)
        return tmp_path / f"{input_path.stem}.html"

    monkeypatch.setattr(cli, "_process_one_file", fake_process)
    result = runner.invoke(cli.app, ["transcribe", "--batch", str(tmp_path)])

    assert result.exit_code == 0
    assert sorted(path.name for path in calls) == ["a.wav", "b.mp3"]


def test_extract_audio_command_uses_default_wav_output(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "meeting.mp4"
    input_path.write_bytes(b"video")
    captured: dict[str, Path | None] = {"input": None, "output": None}

    def fake_extract_audio(
        source: Path, output_path: Path | None = None, console: Console | None = None
    ) -> Path:
        captured["input"] = source
        captured["output"] = output_path
        assert output_path is not None
        output_path.write_bytes(b"wav")
        return output_path

    monkeypatch.setattr(cli, "extract_audio", fake_extract_audio)

    result = runner.invoke(cli.app, ["extract-audio", str(input_path)])

    assert result.exit_code == 0
    assert captured["input"] == input_path
    assert captured["output"] == tmp_path / "meeting.wav"
    assert (tmp_path / "meeting.wav").exists()


def test_extract_audio_command_avoids_overwriting_wav_input(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"wav")
    captured: dict[str, Path | None] = {"output": None}

    def fake_extract_audio(
        source: Path, output_path: Path | None = None, console: Console | None = None
    ) -> Path:
        assert source == input_path
        captured["output"] = output_path
        assert output_path is not None
        output_path.write_bytes(b"normalized wav")
        return output_path

    monkeypatch.setattr(cli, "extract_audio", fake_extract_audio)

    result = runner.invoke(cli.app, ["extract-audio", str(input_path)])

    assert result.exit_code == 0
    assert captured["output"] == tmp_path / "meeting.extracted.wav"
    assert input_path.read_bytes() == b"wav"
    assert (tmp_path / "meeting.extracted.wav").exists()


def test_resolve_audio_output_path_rejects_non_wav_output(tmp_path: Path) -> None:
    input_path = tmp_path / "meeting.mp4"
    input_path.write_bytes(b"video")

    with pytest.raises(typer.BadParameter, match="--output must end with .wav."):
        cli._resolve_audio_output_path(input_path, tmp_path / "meeting.mp3")
