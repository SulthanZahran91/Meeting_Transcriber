# Performance Report

Date: 2026-03-07 (UTC)  
Machine: CPU-only sandbox (no CUDA device detected)  
Sample: `samples/korean_sample.ogg` (1.323537s audio)

## How It Was Measured

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/perf_sample.py
```

Script:

- [`scripts/perf_sample.py`](scripts/perf_sample.py)

## Raw Results

- ASR model load (`tiny`, CPU, int8): `3.862s`
- MT model load (`Helsinki-NLP/opus-mt-ko-en`): `5.487s`
- Transcription runtime: `0.992s`
- Transcription realtime factor: `1.335x` (audio_seconds / runtime_seconds)

Translation throughput (160 short segments):

- Batch `8`: `11.470s` (`13.95 seg/s`)
- Batch `12`: `9.720s` (`16.46 seg/s`)
- Batch `16`: `8.526s` (`18.77 seg/s`)

VAD comparison on this short sample:

- VAD on: `0.992s`
- VAD off: `0.647s`
- Relative delta: `-53.39%` (negative means VAD overhead exceeded savings on this very short clip)

Model/memory behavior:

- ASR model cache reuse: `true`
- RSS delta across 20 repeated transcribe calls: `-6.92 MB` (no growth trend observed in this run)

## Interpretation

- For short clips, VAD can add overhead; it is still expected to help on long meetings with silence.
- Batch size `16` is currently best on this machine for translation throughput.
- Model reuse is functioning correctly.
- No memory growth issue was observed in the repeated short-run check.

## Recommended Defaults (Current Evidence)

- CPU ASR: keep `int8`.
- Translation batch size: keep in `12-16` range; prefer `16` on CPU for throughput.
- Keep VAD enabled by default for long-form meetings, but document that short clips may run faster without it.

## Notes

- Results are sample-size limited (single short clip).
- Re-run this benchmark with 5-10 minute real meeting audio before final production tuning.

