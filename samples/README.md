# Sample Artifact

This folder contains a reproducible sample used to verify end-to-end Korean transcription and translation.

## Files

- `korean_sample.ogg`: Source audio sample.
- `korean_sample.result.md`: Generated output from the current pipeline run.

## Source

- URL: `https://commons.wikimedia.org/wiki/Special:FilePath/Ko-%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD.ogg`
- Retrieved on: `2026-03-07` (UTC)

## Reproduce

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from pathlib import Path; from meeting_transcriber.config import TranscriberConfig; from meeting_transcriber.transcribe import transcribe; from meeting_transcriber.translate import translate_segments; from meeting_transcriber.glossary import apply_glossary; from meeting_transcriber.formatter import format_output; cfg=TranscriberConfig(whisper_model='tiny', device='cpu', compute_type='int8', beam_size=1, output_format='markdown'); segs=transcribe(Path('samples/korean_sample.ogg'), cfg); tr=translate_segments(segs, cfg); tr=apply_glossary(tr, Path('glossary.json')); format_output(tr, 'markdown', Path('output/korean_sample.md'), metadata={'source':'korean_sample.ogg','model':'tiny/cpu/int8','processing_time':'n/a'})"
```

