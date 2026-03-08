# Meeting Transcriber

Offline CLI tool for Korean workplace meeting transcription and English translation.

## Requirements

- Python `>=3.10`
- `uv`
- `ffmpeg` binary available on `PATH`

### Install ffmpeg

Ubuntu/Debian:

```bash
sudo apt update && sudo apt install -y ffmpeg
```

macOS (Homebrew):

```bash
brew install ffmpeg
```

Windows (winget):

```powershell
winget install Gyan.FFmpeg
```

## Install Python dependencies

Base install:

```bash
uv sync
```

Install translation backend (PyTorch):

```bash
uv sync --extra gpu
```

Note: translation currently requires PyTorch. If PyTorch is missing, run with `--no-translate` or install the extra above.
ASR hardware auto-detection checks PyTorch first and falls back to `nvidia-smi` on NVIDIA systems, so `auto` can still pick CUDA for Whisper-only runs.

## Quickstart

Show hardware detection and recommended config:

```bash
uv run meeting-transcriber transcribe --show-hardware
```

Single file:

```bash
uv run meeting-transcriber transcribe ./meeting.mp4
```

Batch directory:

```bash
uv run meeting-transcriber transcribe --batch ./recordings
```

Korean-only transcript:

```bash
uv run meeting-transcriber transcribe ./meeting.mp4 --no-translate
```

## CLI options

```text
--model auto|tiny|small|medium|large-v3
--device auto|cpu|cuda
--compute-type auto|int8|float16|float32
--format html|markdown|srt
--output-dir PATH
--glossary PATH
--no-translate
--batch DIR
--show-hardware
```

## Hardware auto-selection

On startup, the CLI detects RAM/CPU/GPU and resolves `auto` settings:

- CUDA VRAM `>=10GB` -> `large-v3`, `float16`, beam `5`
- CUDA VRAM `6-10GB` -> `medium`, `float16`, beam `5`
- CUDA VRAM `4-6GB` -> `small`, `float16`, beam `5`
- CPU RAM `>=24GB` -> `large-v3`, `int8`, beam `5`
- CPU RAM `12-24GB` -> `medium`, `int8`, beam `5`
- CPU RAM `8-12GB` -> `small`, `int8`, beam `3`
- CPU RAM `<8GB` -> `tiny`, `int8`, beam `1`

Forced model/device options are honored. Risky combinations print warnings.

## Outputs

Default output path is `./output`.

- `html`: side-by-side Korean and English table with metadata header.
- `markdown`: `Time | Korean | English` table.
- `srt`: English subtitles and an additional Korean file with `.ko.srt` suffix.

Example:

```bash
uv run meeting-transcriber transcribe ./meeting.mp4 --format srt
```

## Glossary customization

Edit `glossary.json`:

```json
{
  "corrections": {
    "stoker": "stocker",
    "conveyer": "conveyor"
  },
  "korean_overrides": {
    "반송": "transport/conveyance"
  }
}
```

- `corrections`: case-insensitive English term replacements.
- `korean_overrides`: if Korean text contains a key, the mapped English term is injected.

## Offline behavior

- First run downloads model files (Whisper + translation model) to local cache.
- Subsequent runs are offline if caches are present.
- If downloads fail, retry with network access once, then run offline.

## Troubleshooting

`ffmpeg is not installed`:
- Install ffmpeg and verify `ffmpeg -version` works.

`ASR out of memory`:
- Try `--model small` or `--model tiny`.
- On CUDA, retry with `--device cpu`.

`--show-hardware` does not detect your NVIDIA GPU:
- Verify the NVIDIA driver is installed and `nvidia-smi` works.
- If Whisper GPU inference works but auto-detection is still wrong, use `--device cuda`.

`Translation backend requires PyTorch`:
- Run `uv sync --extra gpu`.
- Or use `--no-translate`.

`Model download failed`:
- Check internet access, retry once, then rerun after cache is populated.
