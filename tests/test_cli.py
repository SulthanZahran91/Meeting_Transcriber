from pathlib import Path

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

