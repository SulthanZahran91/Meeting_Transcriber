from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from meeting_transcriber.config import TranscriberConfig, resolve_auto_config
from meeting_transcriber.extract import extract_audio
from meeting_transcriber.formatter import format_output
from meeting_transcriber.glossary import apply_glossary
from meeting_transcriber.hardware import detect_hardware, print_hardware_summary, recommend_config
from meeting_transcriber.transcribe import Segment, transcribe
from meeting_transcriber.translate import TranslatedSegment, translate_segments

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}


@app.callback()
def main() -> None:
    """Meeting transcription and translation CLI."""


def transcribe_cmd(
    input_file: Optional[Path] = typer.Argument(None, help="Input audio/video file"),
    model: str = typer.Option(
        "auto", "--model", help="Whisper model (auto/tiny/small/medium/large-v3)"
    ),
    device: str = typer.Option("auto", "--device", help="Device (auto/cpu/cuda)"),
    compute_type: str = typer.Option(
        "auto", "--compute-type", help="Compute type (auto/int8/float16/float32)"
    ),
    format: str = typer.Option("html", "--format", help="Output format (html/markdown/srt)"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", help="Output directory"),
    glossary: Path = typer.Option(Path("glossary.json"), "--glossary", help="Glossary path"),
    no_translate: bool = typer.Option(False, "--no-translate", help="Skip English translation"),
    batch: Optional[Path] = typer.Option(
        None, "--batch", help="Process all supported files in a directory"
    ),
    show_hardware: bool = typer.Option(
        False, "--show-hardware", help="Show hardware profile + recommendation and exit"
    ),
) -> None:
    _validate_options(model, device, compute_type, format)
    _validate_input_args(input_file, batch, show_hardware)

    force_model = None if model == "auto" else model
    force_device = None if device == "auto" else device
    hw = detect_hardware()
    recommendation = recommend_config(
        hw=hw, force_model=force_model, force_device=force_device
    )
    print_hardware_summary(hw, recommendation, console=console)
    if "WARNING:" in recommendation.reason:
        console.print(f"[yellow]{recommendation.reason}[/yellow]")

    if show_hardware:
        return

    config = resolve_auto_config(
        TranscriberConfig(
            whisper_model=model,
            device=device,
            compute_type=compute_type,
            output_format=format,
            output_dir=output_dir,
            glossary_path=glossary,
        ),
        recommendation,
    )
    _print_resolved_config(config)

    files = _resolve_target_files(input_file, batch)
    if not files:
        raise typer.Exit(code=1)

    started_all = time.perf_counter()
    successes = 0
    failures = 0

    for target in files:
        try:
            output_path = _process_one_file(target, config, no_translate=no_translate)
            elapsed = time.perf_counter() - started_all
            console.print(
                f"[green]Completed:[/green] {target.name} -> {output_path} ({elapsed:.1f}s total)"
            )
            successes += 1
        except Exception as exc:
            console.print(f"[red]Failed:[/red] {target}: {exc}")
            failures += 1
            if batch is None:
                raise typer.Exit(code=1)

    if batch is not None:
        total_elapsed = time.perf_counter() - started_all
        console.print(
            Panel(
                f"Processed: {len(files)}\nSuccess: {successes}\nFailed: {failures}\nElapsed: {total_elapsed:.1f}s",
                title="Batch Summary",
                expand=False,
            )
        )
        if failures > 0:
            raise typer.Exit(code=1)


def extract_audio_cmd(
    input_file: Path = typer.Argument(..., help="Input audio/video file"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output WAV file path"),
) -> None:
    target = _resolve_target_files(input_file, batch=None)[0]
    output_path = _resolve_audio_output_path(target, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_path = extract_audio(target, output_path=output_path, console=console)
    console.print(f"[green]Extracted audio:[/green] {target.name} -> {final_path}")


def _process_one_file(
    input_path: Path, config: TranscriberConfig, no_translate: bool
) -> Path:
    started = time.perf_counter()
    temp_audio: Optional[Path] = None
    try:
        temp_audio = extract_audio(input_path, output_path=None, console=console)
        segments = _transcribe_with_cuda_fallback(temp_audio, config)
        translated = _translate_pipeline(segments, config, no_translate=no_translate)

        suffix = {"html": ".html", "markdown": ".md", "srt": ".srt"}[config.output_format]
        output_path = config.output_dir / f"{input_path.stem}{suffix}"
        elapsed = time.perf_counter() - started
        metadata = {
            "source": input_path.name,
            "date": datetime.now(timezone.utc).isoformat(),
            "model": f"{config.whisper_model} ({config.device}/{config.compute_type})",
            "processing_time": f"{elapsed:.1f}s",
        }
        final_path = format_output(
            translated,
            config.output_format,
            output_path,
            metadata=metadata,
        )
        console.print(
            f"[cyan]Summary:[/cyan] segments={len(translated)} elapsed={elapsed:.1f}s output={final_path}"
        )
        return final_path
    finally:
        if temp_audio is not None and temp_audio.exists():
            temp_audio.unlink(missing_ok=True)


def _transcribe_with_cuda_fallback(audio_path: Path, config: TranscriberConfig) -> list[Segment]:
    try:
        return transcribe(audio_path, config, console=console)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if config.device == "cuda" and "out of memory" in msg:
            console.print("[yellow]CUDA OOM detected. Retrying transcription on CPU/int8...[/yellow]")
            cpu_config = replace(config, device="cpu", compute_type="int8")
            return transcribe(audio_path, cpu_config, console=console)
        raise


def _translate_pipeline(
    segments: list[Segment], config: TranscriberConfig, no_translate: bool
) -> list[TranslatedSegment]:
    if no_translate:
        return [
            TranslatedSegment(
                start=seg.start,
                end=seg.end,
                korean=seg.text,
                english="",
            )
            for seg in segments
        ]

    translated = translate_segments(segments, config, console=console)
    return apply_glossary(translated, config.glossary_path)


def _validate_options(model: str, device: str, compute_type: str, format: str) -> None:
    valid_models = {"auto", "tiny", "small", "medium", "large-v3"}
    valid_devices = {"auto", "cpu", "cuda"}
    valid_compute = {"auto", "int8", "float16", "float32"}
    valid_formats = {"html", "markdown", "srt"}
    if model not in valid_models:
        raise typer.BadParameter(f"Invalid --model '{model}'.")
    if device not in valid_devices:
        raise typer.BadParameter(f"Invalid --device '{device}'.")
    if compute_type not in valid_compute:
        raise typer.BadParameter(f"Invalid --compute-type '{compute_type}'.")
    if format not in valid_formats:
        raise typer.BadParameter(f"Invalid --format '{format}'.")


def _validate_input_args(
    input_file: Optional[Path], batch: Optional[Path], show_hardware: bool
) -> None:
    if show_hardware:
        return
    if input_file is None and batch is None:
        raise typer.BadParameter("Provide INPUT_FILE or use --batch DIR.")
    if input_file is not None and batch is not None:
        raise typer.BadParameter("Use either INPUT_FILE or --batch, not both.")


def _resolve_target_files(input_file: Optional[Path], batch: Optional[Path]) -> list[Path]:
    if batch is not None:
        if not batch.exists() or not batch.is_dir():
            raise typer.BadParameter(f"--batch must be a directory: {batch}")
        files = sorted(
            path for path in batch.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            console.print(f"[red]No supported media files found in {batch}[/red]")
        return files

    assert input_file is not None
    if not input_file.exists() or not input_file.is_file():
        raise typer.BadParameter(f"Input file not found: {input_file}")
    if input_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(
            f"[yellow]Warning: file extension {input_file.suffix} is not in supported list; attempting anyway.[/yellow]"
        )
    return [input_file]


def _resolve_audio_output_path(input_path: Path, output: Optional[Path]) -> Path:
    if output is None:
        if input_path.suffix.lower() == ".wav":
            return input_path.with_name(f"{input_path.stem}.extracted.wav")
        return input_path.with_suffix(".wav")

    if output.suffix.lower() != ".wav":
        raise typer.BadParameter("--output must end with .wav.")

    if output.resolve() == input_path.resolve():
        raise typer.BadParameter("Refusing to overwrite the input file. Choose a different --output.")

    return output


def _print_resolved_config(config: TranscriberConfig) -> None:
    console.print(
        Panel(
            "\n".join(
                [
                    f"Whisper model: {config.whisper_model}",
                    f"Device: {config.device}",
                    f"Compute type: {config.compute_type}",
                    f"Beam size: {config.beam_size}",
                    f"Language: {config.language}",
                    f"Translation model: {config.translation_model}",
                    f"Output format: {config.output_format}",
                    f"Output dir: {config.output_dir}",
                    f"Glossary path: {config.glossary_path}",
                ]
            ),
            title="Resolved Config",
            expand=False,
        )
    )


app.command(name="transcribe")(transcribe_cmd)
app.command(name="extract-audio")(extract_audio_cmd)


if __name__ == "__main__":
    app()
