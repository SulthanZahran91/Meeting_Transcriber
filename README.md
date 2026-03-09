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

Install optional PyTorch backend:

```bash
uv sync --extra gpu
```

Note: translation currently requires PyTorch. The default translator is `Qwen/Qwen3.5-4B`, so GPU inference and healthy RAM/VRAM are still recommended if you want the accuracy-first default path. If PyTorch is missing, run with `--no-translate` or install the extra above.

### GPU runtime requirements

`uv sync --extra gpu` installs Python-side GPU support such as PyTorch. It does not install the host NVIDIA driver or CUDA runtime libraries required for Whisper GPU inference.

Whisper ASR in this project runs through `faster-whisper`, which uses CTranslate2 for inference. That means Whisper GPU support is separate from the PyTorch translation backend.

For `--device cuda` or `--device auto` to run Whisper on GPU, the machine also needs:

- An NVIDIA GPU supported by CTranslate2.
- An NVIDIA driver compatible with the installed CUDA runtime.
- CUDA 12 runtime libraries available to the OS loader.
- The libraries required by the current `faster-whisper`/CTranslate2 stack:
  `cuBLAS` for CUDA 12 and `cuDNN 9` for CUDA 12.
- On Windows, the Microsoft Visual C++ runtime.
- `nvidia-smi` working if you want hardware auto-detection to discover the GPU reliably.

For this repo's current stack (`faster-whisper` with `ctranslate2 4.7.1` in `uv.lock`), missing DLL errors such as `cublas64_12.dll` usually mean the CUDA 12 runtime/toolkit is missing from the machine, mismatched with the installed driver/runtime stack, or not available on `PATH`.

If you intentionally target older CUDA/cuDNN combinations, you may need to pin an older `ctranslate2` version instead of using the current default stack.

ASR hardware auto-detection checks PyTorch first and falls back to `nvidia-smi` on NVIDIA systems, so `auto` can still pick CUDA for Whisper-only runs. Detection success does not guarantee that all CUDA libraries needed by `faster-whisper` can actually be loaded at runtime. If GPU detection succeeds but CUDA libraries are unavailable at runtime, transcription falls back to CPU/int8.

## Quickstart

Show hardware detection and recommended config:

```bash
uv run meeting-transcriber transcribe --show-hardware
```

Single file:

```bash
uv run meeting-transcriber transcribe ./meeting.mp4
```

The default output format is now `srt`.

Batch directory:

```bash
uv run meeting-transcriber transcribe --batch ./recordings
```

Korean-only transcript:

```bash
uv run meeting-transcriber transcribe ./meeting.mp4 --no-translate
```

Extract audio only:

```bash
uv run meeting-transcriber extract-audio ./meeting.mp4
```

Extract audio for a directory:

```bash
uv run meeting-transcriber extract-audio --batch ./recordings
```

## CLI options

```text
extract-audio INPUT_FILE [--output PATH]
extract-audio --batch DIR

--model auto|tiny|small|medium|large-v3
--device auto|cpu|cuda
--compute-type auto|int8|float16|float32
--format html|markdown|srt
--output-dir PATH
--glossary PATH
--translation-model REPO_OR_PATH
--no-translate
--batch DIR
--show-hardware
```

`extract-audio` writes a 16 kHz mono WAV file. By default it saves next to the input as `<name>.wav`; if the input is already `.wav`, it writes `<name>.extracted.wav` to avoid overwriting the source. In batch mode, it applies the same naming rule to every supported file in the target directory. `--output` is single-file only.

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

Default output path is `./output`. Default output format is `srt`.

- `html`: side-by-side Korean and English table with metadata header.
- `markdown`: `Time | Korean | English` table.
- `srt`: English subtitles and an additional Korean file with `.ko.srt` suffix.

## Translation model

Default translation model: `Qwen/Qwen3.5-4B`

- The translator uses a 7-part context window for each subtitle segment: the target segment, 3 segments before it, and 3 segments after it.
- Translation is generation-based and optimized for accuracy rather than throughput.
- Qwen3.5 thinking mode is disabled in the translation prompt path so the model returns direct subtitle text instead of reasoning traces.
- `transformers>=5.3.0` is required for the current default Qwen3.5 setup.

Example:

```bash
uv run meeting-transcriber transcribe ./meeting.mp4 --format srt
```

Use a different translation checkpoint when the default 4B model is too large or too slow to download:

```bash
uv run meeting-transcriber transcribe ./meeting.mp4 --translation-model Qwen/Qwen2.5-1.5B-Instruct
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

`Library cublas64_12.dll is not found or cannot be loaded`:
- GPU detection succeeded, but the CUDA runtime libraries needed by Whisper were not available at runtime.
- Install or repair the NVIDIA/CUDA 12 runtime on the host machine and ensure the CUDA `bin` directory is on `PATH`.
- If you do not need GPU inference, run with `--device cpu`.

`--show-hardware` does not detect your NVIDIA GPU:
- Verify the NVIDIA driver is installed and `nvidia-smi` works.
- If Whisper GPU inference works but auto-detection is still wrong, use `--device cuda`.

`Translation backend requires PyTorch`:
- Run `uv sync --extra gpu`.
- Or use `--no-translate`.

`Translation model ran out of memory`:
- The default Qwen3.5 translator is still accuracy-first and relatively heavy.
- Use a machine with more RAM/VRAM, or switch to a smaller checkpoint with `--translation-model`.

`Model download failed`:
- Check internet access, retry once, then rerun after cache is populated.
- If the default model is impractical for your machine or network, choose a smaller checkpoint with `--translation-model`.
