from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Optional

import ffmpeg
from rich.console import Console


def extract_audio(
    input_path: Path, output_path: Optional[Path] = None, console: Optional[Console] = None
) -> Path:
    console = console or Console()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. Install ffmpeg and retry."
        )

    temp_created = False
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        output_path = Path(tmp.name)
        temp_created = True

    assert output_path is not None

    try:
        with console.status("Extracting audio (WAV 16kHz mono)..."):
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                ac=1,
                ar=16000,
                format="wav",
                acodec="pcm_s16le",
            )
            ffmpeg.run(
                stream,
                overwrite_output=True,
                capture_stdout=True,
                capture_stderr=True,
                quiet=True,
            )
    except ffmpeg.Error as exc:
        if temp_created and output_path.exists():
            output_path.unlink(missing_ok=True)
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg extraction failed: {stderr}") from exc

    return output_path

