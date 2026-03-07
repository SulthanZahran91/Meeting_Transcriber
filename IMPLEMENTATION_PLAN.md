# Meeting Transcriber/Translator Implementation Plan

Source spec: `spec.md`  
Target: Fully offline Korean meeting transcription + English translation CLI, CPU-first with optional CUDA acceleration.

## 1) Scope and Success Criteria

### Functional Goals
- [ ] Accept video/audio input and normalize audio to WAV 16kHz mono PCM s16le.
- [ ] Produce Korean transcript segments with timestamps.
- [ ] Produce English translations aligned to Korean segments.
- [ ] Apply glossary corrections and Korean keyword overrides.
- [ ] Export output as HTML (default), Markdown, and SRT (+ `.ko.srt` for Korean).
- [ ] Provide Typer CLI with all options defined in `spec.md`.
- [ ] Support single-file and batch directory processing.

### Non-Functional Goals
- [ ] Run fully offline after model download.
- [ ] Run on CPU-only machine (i5-1155G7, 16GB RAM) with reasonable defaults.
- [ ] Auto-detect hardware and resolve `auto` config at startup.
- [ ] Reuse loaded models (avoid repeated model init in batch runs).
- [ ] Provide clear progress and user-facing summaries with Rich.

### Done Criteria (Release Gate)
- [ ] End-to-end run completes on a short Korean sample with default options.
- [ ] `--show-hardware` prints profile + recommendation and exits.
- [ ] Error paths provide clear actionable guidance.
- [ ] Basic test suite passes locally.
- [ ] README documents install, usage, and troubleshooting.

## 2) Project Bootstrap

### Repository Setup
- [x] Initialize project metadata in `pyproject.toml`.
- [ ] Add dependencies exactly as specified:
  - [x] `faster-whisper`
  - [x] `transformers`
  - [x] `sentencepiece`
  - [x] `ffmpeg-python`
  - [x] `rich`
  - [x] `typer`
  - [x] `psutil`
- [x] Add optional dependency group `gpu = ["torch>=2.0.0"]`.
- [x] Add script entrypoint: `meeting-transcriber = "meeting_transcriber.cli:app"`.
- [x] Create package directory structure under `src/meeting_transcriber/`.
- [x] Add `output/` and default `glossary.json`.

### Initial Validation
- [x] `uv run meeting-transcriber --help` works.
- [x] Imports resolve from `src/` package layout.
- [ ] Basic lint/type tooling decision documented (if used).

## 3) Data Models and Shared Config

### Dataclasses and Types
- [x] Implement `HardwareProfile` dataclass in `hardware.py`.
- [x] Implement `ModelRecommendation` dataclass in `hardware.py`.
- [x] Implement `TranscriberConfig` dataclass in `config.py`.
- [x] Implement `Segment` dataclass in `transcribe.py`.
- [x] Implement `TranslatedSegment` dataclass in `translate.py` (or shared types module if refactored).

### Config Resolution
- [x] Support `auto` for `whisper_model`, `device`, `compute_type`.
- [ ] Define a clear config resolution function that:
  - [x] Detects hardware.
  - [x] Resolves recommendations.
  - [x] Applies user overrides with warnings when risky.
- [x] Ensure resolved config is printed once at startup (Rich panel).

### Validation Rules
- [ ] Validate enums/options (`model`, `device`, `compute_type`, `format`).
- [ ] Validate `beam_size > 0`.
- [ ] Validate paths (`output_dir`, `glossary_path`).

## 4) Hardware Detection and Recommendation

### `detect_hardware()`
- [x] Detect CUDA availability via `torch.cuda.is_available()` when torch exists.
- [x] On CUDA: capture GPU name + VRAM GB from device properties.
- [x] On missing torch/no CUDA: fallback cleanly to CPU.
- [x] Capture RAM GB via `psutil.virtual_memory().total`.
- [x] Capture CPU name and core count.
- [x] Return complete `HardwareProfile` object with safe defaults for missing fields.

### `recommend_config()`
- [x] Encode full selection matrix from spec for CPU and CUDA.
- [x] Support `force_model` override logic:
  - [x] Keep forced model.
  - [x] Still auto-pick compatible device/compute/beam when not forced.
  - [x] Warn when likely OOM/risky.
- [x] Support `force_device` override logic:
  - [x] Keep forced device.
  - [x] Recompute model/compute/beam accordingly.
- [x] Support both forced simultaneously and still compute safe `compute_type`.
- [x] Produce human-readable `reason`.
- [x] Render recommendation panel via Rich.

### Test Checklist
- [x] Unit tests for matrix mapping boundaries (exact 4/6/8/10/12/24 GB thresholds).
- [x] Unit tests for override precedence (`force_model`, `force_device`, both).
- [ ] Unit tests for no-torch environment fallback.

## 5) Audio Extraction Module (`extract.py`)

### Implementation
- [x] Validate `ffmpeg` binary exists via `shutil.which("ffmpeg")`.
- [x] Raise clear error with install instructions when missing.
- [x] Convert any input media to WAV 16kHz mono PCM s16le.
- [x] Accept optional output path; create temp file when omitted.
- [x] Return resulting path.
- [x] Capture and surface ffmpeg stderr on failure.
- [x] Show spinner/progress messaging via Rich.

### Test Checklist
- [ ] Unit test: ffmpeg missing error path.
- [ ] Integration test: short sample conversion creates expected WAV parameters.
- [ ] Unit test: temp output path cleanup ownership documented.

## 6) Transcription Module (`transcribe.py`)

### Implementation
- [x] Load `WhisperModel` with resolved config values.
- [x] Call `model.transcribe()` with:
  - [x] `language="ko"`
  - [x] `beam_size=config.beam_size`
  - [x] `vad_filter=config.vad_filter`
  - [x] `vad_parameters={"min_silence_duration_ms": 500}`
- [x] Consume lazy segment generator and collect list of `Segment`.
- [x] Print progress every 50 segments.
- [ ] Handle and classify errors:
  - [ ] model download/load failures
  - [ ] OOM errors
  - [ ] CUDA-specific OOM fallback path (if in CLI orchestration)

### Performance/Reuse
- [x] Design model loader/cache to avoid repeated load in batch mode.
- [ ] Keep API simple for single-file and multi-file reuse.

### Test Checklist
- [x] Unit tests with mocked Whisper output generator.
- [ ] Unit tests for progress logging every 50 segments.
- [ ] Error translation tests for OOM guidance text.

## 7) Translation Module (`translate.py`)

### Implementation
- [x] Load `MarianTokenizer` + `MarianMTModel` from `config.translation_model`.
- [x] Batch segments in size range 8-16 (configurable constant).
- [x] Skip empty text safely while preserving timestamps.
- [x] Decode and map back to `TranslatedSegment`.
- [x] Rich progress bar over total segments.

### Performance/Reuse
- [x] Reuse tokenizer/model across batch files.
- [x] Avoid per-segment inference.
- [x] Keep memory usage bounded by batch size.

### Test Checklist
- [ ] Unit tests for batching logic (exact batch boundaries).
- [ ] Unit tests for empty/whitespace segments.
- [ ] Unit tests ensuring timestamp alignment preserved.

## 8) Glossary Module (`glossary.py`)

### Implementation
- [x] Load `glossary.json` safely with clear error on malformed JSON.
- [x] Apply `corrections` on English text using case-insensitive word boundaries.
- [x] Apply `korean_overrides` based on Korean text containment.
- [x] Apply replacements without mutating input objects.
- [x] Return new list of `TranslatedSegment`.

### Edge Cases
- [ ] Missing glossary file: decide behavior (warn + skip vs hard fail) and document.
- [ ] Empty/missing keys in glossary sections handled gracefully.
- [ ] Ensure acronym casing outcomes are deterministic.

### Test Checklist
- [x] Unit tests for case-insensitive corrections.
- [x] Unit tests for word-boundary safety (no partial-word corruption).
- [ ] Unit tests for Korean override precedence over base translation text.
- [x] Unit tests verifying immutability (input unchanged).

## 9) Formatter Module (`formatter.py`)

### Shared Formatting
- [x] Implement timestamp helpers:
  - [x] `HH:MM:SS.mmm` for html/markdown
  - [x] `HH:MM:SS,mmm` for srt
- [x] Ensure output directory creation.
- [x] Return written output path.

### HTML Output
- [x] Two-column readable table:
  - [x] left: timestamp + Korean
  - [x] right: English
- [x] Inline CSS with readability features:
  - [x] monospace timestamps
  - [x] alternating row colors
  - [x] responsive width behavior
- [x] Metadata header includes:
  - [x] source filename
  - [x] run date/time
  - [x] model used
  - [x] processing time

### Markdown Output
- [x] Render table `Time | Korean | English`.
- [x] Escape pipes/newlines safely.

### SRT Output
- [x] Generate English SRT as primary output.
- [x] Generate Korean SRT with `.ko.srt` suffix.
- [x] Ensure sequential numbering and valid timing format.

### Test Checklist
- [x] Unit tests for timestamp formatting (including millisecond rounding).
- [x] Golden-file tests for html/markdown/srt outputs.
- [x] Verify unicode Korean text is preserved.

## 10) CLI Orchestration (`cli.py`)

### Command Surface
- [x] Implement Typer app with main `transcribe` command.
- [x] Support required options:
  - [x] `--model`
  - [x] `--device`
  - [x] `--compute-type`
  - [x] `--format`
  - [x] `--output-dir`
  - [x] `--glossary`
  - [x] `--no-translate`
  - [x] `--batch`
  - [x] `--show-hardware`

### Runtime Flow
- [x] Detect hardware and resolve config.
- [x] Print recommendation/config summary panel.
- [x] Validate input path(s) before heavy model load.
- [x] Extract audio with spinner.
- [x] Run transcription and translation with progress.
- [x] Apply glossary.
- [x] Format output and write files.
- [x] Print final summary:
  - [x] total segments
  - [x] elapsed processing time
  - [x] output path
- [x] Clean temp audio files reliably (`finally`).

### Batch Processing
- [x] Discover supported media files in directory.
- [x] Process sequentially.
- [x] Reuse loaded ASR and translation models across files.
- [x] Per-file success/failure reporting and final batch summary.

### Test Checklist
- [x] CLI help snapshot test.
- [x] Single-file command test with mocked internals.
- [x] `--show-hardware` exit-path test.
- [ ] Batch mode test verifying one-time model load.

## 11) Error Handling and Recovery

### Required Error Cases
- [x] ffmpeg missing: actionable install message.
- [x] Model download failure: retry/manual-download guidance.
- [x] ASR/translation OOM: suggest smaller model (`small`/`tiny`).
- [x] CUDA OOM: warn and retry on CPU automatically.
- [x] Invalid input path/empty batch dir: clear message and non-zero exit.
- [x] Invalid format/model option: friendly validation error.

### Observability
- [ ] Distinguish warning vs error output levels with Rich styling.
- [ ] Include root exception context without stack trace noise by default.
- [ ] Optional verbose/debug mode decision documented.

## 12) Testing Strategy

### Unit Tests
- [x] Hardware detection/recommendation logic (mocked environments).
- [x] Config resolution and override precedence.
- [x] Glossary replacement correctness and immutability.
- [x] Timestamp formatting and output writers.

### Integration Tests
- [x] Audio extraction integration with short fixture media.
- [x] CLI smoke test through full pipeline with mocked models.
- [x] Batch processing integration with 2+ fixture files.

### Regression Fixtures
- [x] Add small Korean sample transcript fixture.
- [x] Add representative glossary fixture.
- [x] Add expected output snapshots for html/md/srt.

### Test Execution
- [x] Define one command for local validation (`uv run ...`).
- [x] Ensure tests pass on CPU-only environment.

## 13) Performance Validation

### Baseline Measurements
- [x] Measure model load time (ASR + MT).
- [ ] Measure transcription throughput on 5-10 minute sample.
- [x] Measure translation throughput with batch sizes 8, 12, 16.
- [x] Validate VAD impact on runtime.

### Tuning Decisions
- [x] Select default translation batch size for CPU balance.
- [ ] Confirm CPU `int8` defaults avoid OOM on 16GB RAM.
- [ ] Validate GPU `float16` path where available.

### Acceptance Targets
- [ ] Runtime behavior is within expected spec ranges.
- [x] No repeated model loads in batch mode.
- [x] No unbounded memory growth in long runs.

## 14) Documentation and UX

### README Content
- [x] Installation with `uv`.
- [x] ffmpeg prerequisite and install instructions by OS.
- [x] Quickstart examples (single file and batch).
- [x] Hardware auto-selection explanation.
- [x] Troubleshooting section (OOM, CUDA issues, model download).
- [x] Output examples (html/md/srt).

### User Assets
- [x] Provide default `glossary.json` with sample corrections.
- [x] Document how to customize glossary safely.
- [x] Clarify offline behavior after first model download.

## 15) Delivery Sequence (Recommended Order)

- [x] Milestone 1: Bootstrap + dataclasses + config resolution.
- [x] Milestone 2: Hardware detection/recommendation + Rich panels.
- [x] Milestone 3: Audio extraction + error surfaces.
- [x] Milestone 4: Transcription pipeline (single-file).
- [x] Milestone 5: Translation batching + progress.
- [x] Milestone 6: Glossary + output formatters.
- [x] Milestone 7: CLI orchestration end-to-end.
- [x] Milestone 8: Batch mode + model reuse.
- [ ] Milestone 9: Tests + docs + performance validation.

## 16) Risk Register and Mitigations

- [ ] Risk: slow/unstable first model download.
  - Mitigation: clear retry guidance and cache location notes.
- [ ] Risk: CPU OOM on large models.
  - Mitigation: conservative auto defaults (`int8`, smaller model) + warnings.
- [ ] Risk: CUDA OOM mid-run.
  - Mitigation: automatic CPU fallback and resume/retry behavior.
- [ ] Risk: translation quality for domain terms.
  - Mitigation: glossary overrides + iterative glossary tuning process.
- [ ] Risk: ffmpeg environment inconsistencies.
  - Mitigation: explicit binary check and stderr surfacing.
- [ ] Risk: long-run operational failures in batch mode.
  - Mitigation: per-file isolation, continue-on-error option, final summary.

## 17) Final Acceptance Checklist

- [ ] All module-level checklists completed.
- [ ] End-to-end demo executed and verified.
- [ ] Test suite green.
- [ ] Documentation complete and accurate.
- [ ] Known limitations documented.
- [ ] Ready for first user trial.
