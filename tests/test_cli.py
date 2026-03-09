from pathlib import Path

import pytest
from rich.console import Console
import typer
from typer.testing import CliRunner

from meeting_transcriber import cli
from meeting_transcriber.config import DEFAULT_OUTPUT_FORMAT, DEFAULT_TRANSLATION_MODEL, TranscriberConfig
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
        assert config.output_format == DEFAULT_OUTPUT_FORMAT
        assert config.translation_model == DEFAULT_TRANSLATION_MODEL
        return tmp_path / "meeting.html"

    monkeypatch.setattr(cli, "_process_one_file", fake_process)
    result = runner.invoke(cli.app, ["transcribe", str(in_file)])

    assert result.exit_code == 0
    assert calls["count"] == 1


def test_single_file_allows_translation_model_override(monkeypatch, tmp_path: Path) -> None:
    in_file = tmp_path / "meeting.wav"
    in_file.write_bytes(b"wav")
    captured: dict[str, object] = {}

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
        captured["input_path"] = input_path
        captured["translation_model"] = config.translation_model
        return tmp_path / "meeting.html"

    monkeypatch.setattr(cli, "_process_one_file", fake_process)
    result = runner.invoke(
        cli.app,
        ["transcribe", str(in_file), "--translation-model", "Qwen/Qwen2.5-1.5B-Instruct"],
    )

    assert result.exit_code == 0
    assert captured == {
        "input_path": in_file,
        "translation_model": "Qwen/Qwen2.5-1.5B-Instruct",
    }


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


def test_transcribe_with_cuda_fallback_retries_on_missing_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    attempts: list[tuple[str, str]] = []
    expected = [cli.Segment(start=0.0, end=1.0, text="hello")]
    test_console = Console(record=True)

    def fake_transcribe(
        _audio_path: Path, config: TranscriberConfig, console: Console | None = None
    ) -> list[cli.Segment]:
        attempts.append((config.device, config.compute_type))
        if config.device == "cuda":
            raise RuntimeError("ASR failed: Library cublas64_12.dll is not found or cannot be loaded")
        return expected

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    monkeypatch.setattr(cli, "console", test_console)

    segments, runtime_config = cli._transcribe_with_cuda_fallback(
        audio,
        TranscriberConfig(
            whisper_model="small",
            device="cuda",
            compute_type="float16",
            beam_size=5,
        ),
    )

    rendered = test_console.export_text()
    assert segments == expected
    assert runtime_config.device == "cpu"
    assert runtime_config.compute_type == "int8"
    assert attempts == [("cuda", "float16"), ("cpu", "int8")]
    assert "CUDA runtime/library issue detected." in rendered


def test_transcribe_with_cuda_fallback_reraises_unrelated_runtime_error(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")

    def fake_transcribe(
        _audio_path: Path, _config: TranscriberConfig, console: Console | None = None
    ) -> list[cli.Segment]:
        raise RuntimeError("ASR failed: input stream is corrupted")

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)

    with pytest.raises(RuntimeError, match="input stream is corrupted"):
        cli._transcribe_with_cuda_fallback(
            audio,
            TranscriberConfig(
                whisper_model="small",
                device="cuda",
                compute_type="float16",
                beam_size=5,
            ),
        )


def test_process_one_file_uses_runtime_config_after_cuda_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    input_path = tmp_path / "meeting.mp4"
    input_path.write_bytes(b"video")
    temp_audio = tmp_path / "meeting.wav"
    temp_audio.write_bytes(b"wav")
    captured: dict[str, object] = {}

    def fake_extract_audio(
        _input_path: Path, output_path: Path | None = None, console: Console | None = None
    ) -> Path:
        return temp_audio

    def fake_transcribe(
        _audio_path: Path, config: TranscriberConfig, console: Console | None = None
    ) -> list[cli.Segment]:
        if config.device == "cuda":
            raise RuntimeError("ASR failed: Library cublas64_12.dll is not found or cannot be loaded")
        return [cli.Segment(start=0.0, end=1.0, text="hello")]

    def fake_postprocess(
        segments: list[cli.Segment], max_segment_length: int
    ) -> list[cli.Segment]:
        captured["max_segment_length"] = max_segment_length
        return segments

    def fake_translate(
        segments: list[cli.Segment], config: TranscriberConfig, no_translate: bool
    ) -> list[object]:
        captured["translate_device"] = config.device
        captured["translate_compute_type"] = config.compute_type
        captured["no_translate"] = no_translate
        return []

    def fake_format_output(
        translated: list[object],
        output_format: str,
        output_path: Path,
        metadata: dict[str, str],
    ) -> Path:
        captured["translated"] = translated
        captured["output_format"] = output_format
        captured["output_path"] = output_path
        captured["metadata"] = metadata
        return output_path

    monkeypatch.setattr(cli, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    monkeypatch.setattr(cli, "postprocess_segments", fake_postprocess)
    monkeypatch.setattr(cli, "_translate_pipeline", fake_translate)
    monkeypatch.setattr(cli, "format_output", fake_format_output)

    output_path = cli._process_one_file(
        input_path,
        TranscriberConfig(
            whisper_model="small",
            device="cuda",
            compute_type="float16",
            output_format="html",
            output_dir=tmp_path,
        ),
        no_translate=True,
    )

    assert output_path == tmp_path / "meeting.html"
    assert captured["translate_device"] == "cpu"
    assert captured["translate_compute_type"] == "int8"
    assert captured["metadata"] == {
        "source": "meeting.mp4",
        "date": captured["metadata"]["date"],
        "model": "small (cpu/int8)",
        "processing_time": captured["metadata"]["processing_time"],
    }
    assert not temp_audio.exists()


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


def test_extract_audio_command_batch_processes_all_files(
    monkeypatch, tmp_path: Path
) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.wav"
    first.write_bytes(b"video")
    second.write_bytes(b"wav")
    captured: list[tuple[Path, Path]] = []

    def fake_extract_audio(
        source: Path, output_path: Path | None = None, console: Console | None = None
    ) -> Path:
        assert output_path is not None
        captured.append((source, output_path))
        output_path.write_bytes(b"wav")
        return output_path

    monkeypatch.setattr(cli, "extract_audio", fake_extract_audio)

    result = runner.invoke(cli.app, ["extract-audio", "--batch", str(tmp_path)])

    assert result.exit_code == 0
    assert captured == [
        (first, tmp_path / "a.wav"),
        (second, tmp_path / "b.extracted.wav"),
    ]


def test_extract_audio_command_batch_continues_after_failure(
    monkeypatch, tmp_path: Path
) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp3"
    first.write_bytes(b"video")
    second.write_bytes(b"audio")
    calls: list[Path] = []

    def fake_extract_audio(
        source: Path, output_path: Path | None = None, console: Console | None = None
    ) -> Path:
        calls.append(source)
        assert output_path is not None
        if source == first:
            raise RuntimeError("boom")
        output_path.write_bytes(b"wav")
        return output_path

    monkeypatch.setattr(cli, "extract_audio", fake_extract_audio)

    result = runner.invoke(cli.app, ["extract-audio", "--batch", str(tmp_path)])

    assert result.exit_code == 1
    assert calls == [first, second]


def test_extract_audio_command_rejects_output_with_batch(tmp_path: Path) -> None:
    input_path = tmp_path / "meeting.mp4"
    input_path.write_bytes(b"video")
    output_path = tmp_path / "meeting.wav"

    result = runner.invoke(
        cli.app,
        ["extract-audio", "--batch", str(tmp_path), "--output", str(output_path)],
    )

    assert result.exit_code == 2
    assert "Use --output only with a single INPUT_FILE." in result.output


def test_resolve_audio_output_path_rejects_non_wav_output(tmp_path: Path) -> None:
    input_path = tmp_path / "meeting.mp4"
    input_path.write_bytes(b"video")

    with pytest.raises(typer.BadParameter, match="--output must end with .wav."):
        cli._resolve_audio_output_path(input_path, tmp_path / "meeting.mp3")
