# Meeting Transcriber/Translator — Implementation Spec

## Overview
CLI tool that takes a video/audio file of a Korean workplace meeting and produces a side-by-side Korean transcription + English translation. Runs fully offline on CPU (i5-1155G7, 16GB RAM). Managed with `uv`.

## Project Setup

```bash
uv init meeting-transcriber
cd meeting-transcriber
uv add faster-whisper transformers sentencepiece ffmpeg-python rich typer
```

Use `uv run` for all execution. No virtualenv activation needed.

## Project Structure

```
meeting-transcriber/
├── pyproject.toml
├── README.md
├── src/
│   └── meeting_transcriber/
│       ├── __init__.py
│       ├── cli.py            # Typer CLI entry point
│       ├── hardware.py       # Hardware detection & model auto-selection
│       ├── extract.py        # Audio extraction via ffmpeg
│       ├── transcribe.py     # Korean ASR via faster-whisper
│       ├── translate.py      # ko→en translation via Helsinki-NLP
│       ├── glossary.py       # Domain-specific term correction
│       ├── formatter.py      # Output formatting (HTML, Markdown, SRT)
│       └── config.py         # Shared configuration & defaults
├── glossary.json             # User-editable technical term mappings
└── output/                   # Default output directory
```

## Module Specifications

### `hardware.py`
- Function: `detect_hardware() -> HardwareProfile`
- `HardwareProfile` dataclass:
  - `device`: str ("cuda", "cpu")
  - `gpu_name`: str | None
  - `gpu_vram_gb`: float | None
  - `ram_gb`: float
  - `cpu_name`: str
  - `cpu_cores`: int
- Detection logic:
  - Try `import torch; torch.cuda.is_available()` — if True, get GPU name and VRAM via `torch.cuda.get_device_properties(0)`
  - If torch not installed or no CUDA: fallback to CPU
  - RAM: `psutil.virtual_memory().total`
  - CPU: `platform.processor()`, `os.cpu_count()`
- Function: `recommend_config(hw: HardwareProfile, force_model: str | None, force_device: str | None) -> ModelRecommendation`
- `ModelRecommendation` dataclass:
  - `whisper_model`: str
  - `device`: str
  - `compute_type`: str
  - `beam_size`: int
  - `reason`: str (human-readable explanation of why this config was chosen)
- Selection matrix (applied ONLY when no manual override):

  | Device | VRAM / RAM         | Model     | compute_type | beam_size |
  |--------|--------------------|-----------|--------------|-----------|
  | CUDA   | VRAM >= 10GB       | large-v3  | float16      | 5         |
  | CUDA   | 6GB <= VRAM < 10GB | medium    | float16      | 5         |
  | CUDA   | 4GB <= VRAM < 6GB  | small     | float16      | 5         |
  | CUDA   | VRAM < 4GB         | small     | int8         | 3         |
  | CPU    | RAM >= 24GB        | large-v3  | int8         | 5         |
  | CPU    | 12GB <= RAM < 24GB | medium    | int8         | 5         |
  | CPU    | 8GB <= RAM < 12GB  | small     | int8         | 3         |
  | CPU    | RAM < 8GB          | tiny      | int8         | 1         |

- If `force_model` is provided: use that model, but still auto-select device/compute_type/beam_size appropriately for the hardware (warn if the forced model is likely to OOM)
- If `force_device` is provided: use that device, auto-select model/compute_type accordingly
- If both forced: honor both, still auto-select compute_type, print warning if risky
- Always print the recommendation + reason to console via rich panel on startup

### `config.py`
- Dataclass `TranscriberConfig` with fields:
  - `whisper_model`: str = "auto" (options: "auto", "tiny", "small", "medium", "large-v3")
  - `device`: str = "auto" (options: "auto", "cpu", "cuda")
  - `compute_type`: str = "auto" (options: "auto", "int8", "float16", "float32")
  - `beam_size`: int = 5
  - `language`: str = "ko"
  - `translation_model`: str = "Helsinki-NLP/opus-mt-ko-en"
  - `output_format`: str = "html" (options: "html", "markdown", "srt")
  - `output_dir`: Path = Path("output")
  - `glossary_path`: Path = Path("glossary.json")
  - `vad_filter`: bool = True (faster-whisper's Silero VAD — filters silence, speeds up processing)
  - `max_segment_length`: int = 500 (chars, for translation chunking)
- When any field is "auto", resolve via `hardware.recommend_config()` at runtime

### `extract.py`
- Function: `extract_audio(input_path: Path, output_path: Path | None = None) -> Path`
- Uses ffmpeg to extract audio from any video format
- Output: WAV, 16kHz, mono, PCM s16le
- If input is already audio (wav/mp3/flac/ogg), still convert to ensure correct format
- If `output_path` is None, write to temp file and return path
- Raise clear error if ffmpeg is not installed (check with shutil.which)
- Show progress with rich

### `transcribe.py`
- Function: `transcribe(audio_path: Path, config: TranscriberConfig) -> list[Segment]`
- `Segment` is a dataclass: `start: float, end: float, text: str`
- Use `faster_whisper.WhisperModel` with:
  - `model_size_or_path=config.whisper_model`
  - `device=config.device`
  - `compute_type=config.compute_type`
- Call `model.transcribe()` with:
  - `language="ko"`
  - `beam_size=config.beam_size`
  - `vad_filter=config.vad_filter`
  - `vad_parameters=dict(min_silence_duration_ms=500)`
- The transcribe generator is lazy — iterate and collect into list
- Log progress: print segment count every 50 segments using rich
- IMPORTANT: Model loading is slow (~30s). Load once, not per-call.

### `translate.py`
- Function: `translate_segments(segments: list[Segment], config: TranscriberConfig) -> list[TranslatedSegment]`
- `TranslatedSegment` is a dataclass: `start: float, end: float, korean: str, english: str`
- Use `transformers.MarianMTModel` + `MarianTokenizer` from `config.translation_model`
- Batch translation for efficiency:
  - Collect segments into batches of 8-16 sentences
  - Tokenize batch, run model, decode
  - This is significantly faster than one-by-one on CPU
- Handle empty segments gracefully (skip, keep timestamp)
- Show rich progress bar over total segments

### `glossary.py`
- Function: `apply_glossary(segments: list[TranslatedSegment], glossary_path: Path) -> list[TranslatedSegment]`
- Load `glossary.json` — format:
  ```json
  {
    "corrections": {
      "stoker": "stocker",
      "conveyer": "conveyor",
      "OHT": "OHT",
      "amhs": "AMHS",
      "ochang": "Ochang"
    },
    "korean_overrides": {
      "스토커": "stocker",
      "반송": "transport/conveyance"
    }
  }
  ```
- `corrections`: applied to English translations (case-insensitive word boundary replacement)
- `korean_overrides`: if Korean text contains key, force that English translation for the term
- Apply to both `korean` and `english` fields as appropriate
- Return new list (don't mutate in place)

### `formatter.py`
- Function: `format_output(segments: list[TranslatedSegment], format: str, output_path: Path) -> Path`
- **HTML format** (default):
  - Clean, readable two-column table
  - Left column: timestamp + Korean text
  - Right column: English translation
  - Include basic CSS for readability (monospace timestamps, alternating row colors)
  - Include a `<style>` block, no external dependencies
  - Add metadata header: source filename, date, model used, processing time
- **Markdown format**:
  - Table with columns: Time | Korean | English
  - Timestamps formatted as `HH:MM:SS`
- **SRT format**:
  - Standard SRT with English translation as the subtitle text
  - Korean in a second .srt file (same name with `.ko.srt` suffix)
- All formats: timestamps formatted from float seconds to `HH:MM:SS.mmm` or `HH:MM:SS,mmm` (SRT)

### `cli.py`
- Use **Typer** for CLI
- Main command: `transcribe`
  ```
  uv run meeting-transcriber transcribe INPUT_FILE [OPTIONS]
  ```
- Options:
  - `--model`: whisper model size (auto/tiny/small/medium/large-v3), default auto
  - `--device`: force device (auto/cpu/cuda), default auto
  - `--compute-type`: force compute type (auto/int8/float16/float32), default auto
  - `--format`: output format (html/markdown/srt), default html
  - `--output-dir`: output directory, default ./output
  - `--glossary`: path to glossary.json, default ./glossary.json
  - `--no-translate`: skip translation, output Korean-only transcript
  - `--batch`: accept a directory, process all video/audio files in it
  - `--show-hardware`: print detected hardware profile and recommended config, then exit
- Flow:
  1. Detect hardware, resolve "auto" values, print config summary as rich panel
  2. If forced model is risky for detected hardware, print yellow warning but proceed
  3. Validate input file exists
  2. Print config summary with rich
  3. Extract audio (show spinner)
  4. Transcribe (show progress)
  5. Translate (show progress bar)
  6. Apply glossary
  7. Format and write output
  8. Print summary: total segments, processing time, output path
- Clean up temp audio file after processing
- For `--batch`: process files sequentially, load models ONCE, reuse across files

## pyproject.toml

```toml
[project]
name = "meeting-transcriber"
version = "0.1.0"
description = "Offline Korean meeting transcriber and translator"
requires-python = ">=3.10"
dependencies = [
    "faster-whisper>=1.0.0",
    "transformers>=4.36.0",
    "sentencepiece>=0.1.99",
    "ffmpeg-python>=0.2.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
    "psutil>=5.9.0",
]

[project.optional-dependencies]
gpu = ["torch>=2.0.0"]

[project.scripts]
meeting-transcriber = "meeting_transcriber.cli:app"
```

## Performance Notes
- Hardware auto-detection runs at startup and selects the best model/quantization for the machine
- On GPU: float16 is preferred, massive speedup (10-20x over CPU for large-v3)
- On CPU: **int8 quantization** is critical — without it, medium model will OOM or crawl on 16GB RAM
- **VAD filter** typically cuts processing time 20-40% by skipping silence
- **Batch translation** (8-16 segments) is 3-5x faster than segment-by-segment
- Expected processing time for 2hr meeting (CPU i5, 16GB, medium/int8): ~40-80 minutes
- Expected processing time for 2hr meeting (GPU 8GB VRAM, medium/float16): ~5-15 minutes
- Model loading: ~30s for Whisper medium on CPU, ~10s for translation model. Load once.
- `--show-hardware` lets users verify detection before committing to a long run

## Error Handling
- ffmpeg not found: clear message with install instructions
- Model download fails: catch and suggest manual download or retry
- Audio extraction fails: show ffmpeg stderr
- Out of memory: catch, suggest running with `--model small` or `--model tiny`
- CUDA out of memory: catch, fallback to CPU automatically with warning, re-run
- Corrupt/unsupported input file: validate with ffmpeg probe before processing
- torch not installed but CUDA requested: clear message to install gpu extras (`uv add meeting-transcriber[gpu]`)

## First Run
On first execution, models will be downloaded automatically:
- faster-whisper medium (int8): ~800MB
- Helsinki-NLP/opus-mt-ko-en: ~300MB
Subsequent runs use cached models.

## Future Enhancements (don't implement yet)
- Speaker diarization (pyannote.audio)
- Real-time streaming mode
- GUI with Textual
- Configurable source/target language pairs
- LLM-powered post-editing of translations
