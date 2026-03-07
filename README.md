# Meeting Transcriber

Offline CLI tool for Korean workplace meeting transcription and English translation.

## Install

```bash
uv sync
```

Optional GPU extras:

```bash
uv sync --extra gpu
```

`ffmpeg` must be installed and available on `PATH`.

## Usage

Single file:

```bash
uv run meeting-transcriber transcribe ./meeting.mp4
```

Batch directory:

```bash
uv run meeting-transcriber transcribe --batch ./recordings
```

Hardware preview:

```bash
uv run meeting-transcriber transcribe --show-hardware
```

