from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from meeting_transcriber.config import TranscriberConfig
from meeting_transcriber.transcribe import Segment


@dataclass(frozen=True)
class TranslatedSegment:
    start: float
    end: float
    korean: str
    english: str


_TRANSLATION_CACHE: dict[str, tuple[object, object]] = {}
DEFAULT_BATCH_SIZE = 1
_OUTPUT_PREFIX_RE = re.compile(r"^(translation|english|target)\s*:\s*", re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_SYSTEM_PROMPT = (
    "You are an expert Korean-to-English subtitle translator for workplace meetings. "
    "Use nearby subtitle context only to disambiguate meaning, pronouns, ellipsis, and domain terms. "
    "Translate only the TARGET segment into natural English subtitle text. "
    "Return only the English translation for TARGET with no notes, labels, quotes, or alternatives."
)


def translate_segments(
    segments: list[Segment],
    config: TranscriberConfig,
    console: Optional[Console] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[TranslatedSegment]:
    console = console or Console()
    if not segments:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.translation_context_window < 0:
        raise ValueError("translation_context_window must be >= 0")
    if config.translation_max_new_tokens < 1:
        raise ValueError("translation_max_new_tokens must be >= 1")

    tokenizer, model = _load_translation_model(config, console=console)

    translated: list[Optional[TranslatedSegment]] = [None] * len(segments)
    active_indices: list[int] = []
    for idx, seg in enumerate(segments):
        if seg.text.strip():
            active_indices.append(idx)
        else:
            translated[idx] = TranslatedSegment(seg.start, seg.end, seg.text, "")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Translating", total=len(segments))
        completed_non_empty = 0
        for idx in range(0, len(active_indices), batch_size):
            chunk_indexes = active_indices[idx : idx + batch_size]
            chunk_messages = [
                _build_translation_messages(
                    segments,
                    target_index=original_idx,
                    context_window=config.translation_context_window,
                )
                for original_idx in chunk_indexes
            ]
            decoded = _run_translation_batch(
                tokenizer,
                model,
                chunk_messages,
                max_new_tokens=config.translation_max_new_tokens,
            )
            for original_idx, english in zip(chunk_indexes, decoded, strict=True):
                seg = segments[original_idx]
                translated[original_idx] = TranslatedSegment(
                    start=seg.start,
                    end=seg.end,
                    korean=seg.text,
                    english=english,
                )
            completed_non_empty += len(chunk_indexes)
            progress.advance(task, len(chunk_indexes))

        progress.advance(task, len(segments) - completed_non_empty)

    return [item for item in translated if item is not None]


def _build_translation_messages(
    segments: list[Segment],
    target_index: int,
    context_window: int,
) -> list[dict[str, str]]:
    start = max(0, target_index - context_window)
    end = min(len(segments), target_index + context_window + 1)

    context_lines: list[str] = []
    for idx in range(start, end):
        offset = idx - target_index
        label = "TARGET" if offset == 0 else f"CTX{offset:+d}"
        text = segments[idx].text.strip() or "[blank]"
        context_lines.append(f"{label}: {text}")

    user_prompt = "\n".join(
        [
            "Translate the TARGET Korean meeting subtitle into English.",
            "Use the surrounding segments only as context.",
            "Return only the English translation of TARGET.",
            "",
            "Context window:",
            *context_lines,
        ]
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _load_translation_model(
    config: TranscriberConfig, console: Console
) -> tuple[object, object]:
    cached = _TRANSLATION_CACHE.get(config.translation_model)
    if cached is not None:
        return cached

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Failed to import transformers. Install dependencies with `uv sync`."
        ) from exc

    console.print(
        "[cyan]If the translation model is not cached locally yet, the first run will "
        "download it from Hugging Face. Larger checkpoints can take several minutes.[/cyan]"
    )
    with console.status(f"Loading translation model '{config.translation_model}'..."):
        try:
            tokenizer, model = _load_translation_model_once(
                model_name=config.translation_model,
                tokenizer_cls=AutoTokenizer,
                model_cls=AutoModelForCausalLM,
                console=console,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Translation backend requires PyTorch. Install it with "
                "`uv sync --extra gpu` and retry."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                _humanize_translation_error(
                    exc=exc,
                    model_name=config.translation_model,
                )
            ) from exc
        _place_translation_model(model)
        model.eval()

    _ensure_tokenizer_padding(tokenizer)
    _TRANSLATION_CACHE[config.translation_model] = (tokenizer, model)
    return tokenizer, model


def _load_translation_model_once(
    model_name: str,
    tokenizer_cls: object,
    model_cls: object,
    console: Console,
) -> tuple[object, object]:
    tokenizer_kwargs = {"trust_remote_code": True}
    model_kwargs = _translation_model_load_kwargs()

    try:
        tokenizer = tokenizer_cls.from_pretrained(model_name, **tokenizer_kwargs)
        model = model_cls.from_pretrained(model_name, **model_kwargs)
        return tokenizer, model
    except Exception as exc:
        if not _is_hf_snapshot_cache_error(exc):
            raise

        model_dir = _translation_model_dir(model_name)
        if _has_local_model_artifacts(model_dir):
            console.print(
                f"[yellow]Using local translation model cache at {model_dir} (offline fallback).[/yellow]"
            )
            tokenizer = tokenizer_cls.from_pretrained(
                str(model_dir),
                local_files_only=True,
                **tokenizer_kwargs,
            )
            model = model_cls.from_pretrained(
                str(model_dir),
                local_files_only=True,
                **model_kwargs,
            )
            return tokenizer, model

        console.print(
            "[yellow]Detected stale translation model cache. "
            "Refreshing Hugging Face snapshot and retrying once...[/yellow]"
        )
        _refresh_hf_snapshot(model_name=model_name)

        model_dir = _translation_model_dir(model_name)
        tokenizer = tokenizer_cls.from_pretrained(
            str(model_dir),
            local_files_only=True,
            **tokenizer_kwargs,
        )
        model = model_cls.from_pretrained(
            str(model_dir),
            local_files_only=True,
            **model_kwargs,
        )
        return tokenizer, model


def _translation_model_load_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {"trust_remote_code": True}
    dtype = _preferred_torch_dtype()
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def _preferred_torch_dtype() -> object | None:
    try:
        import torch
    except Exception:
        return None

    if torch.cuda.is_available():
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported) and is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    return torch.float32


def _ensure_tokenizer_padding(tokenizer: object) -> None:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if pad_token_id is None and eos_token_id is not None:
        if getattr(tokenizer, "pad_token", None) is None and eos_token is not None:
            setattr(tokenizer, "pad_token", eos_token)
        setattr(tokenizer, "pad_token_id", eos_token_id)


def _place_translation_model(model: object) -> None:
    move_to = getattr(model, "to", None)
    if not callable(move_to):
        return

    try:
        import torch
    except Exception:
        return

    if torch.cuda.is_available():
        move_to("cuda")


def _refresh_hf_snapshot(model_name: str) -> None:
    model_dir = _translation_model_dir(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return
    snapshot_download(
        repo_id=model_name,
        local_dir=str(model_dir),
        force_download=True,
        local_files_only=False,
    )


def _translation_model_dir(model_name: str) -> Path:
    safe_name = model_name.replace("/", "--")
    return Path.home() / ".cache" / "meeting-transcriber" / "models" / safe_name


def _has_local_model_artifacts(model_dir: Path) -> bool:
    return model_dir.exists() and any(model_dir.iterdir())


def _is_hf_snapshot_cache_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return (
        "snapshot folder" in lowered
        or (
            "locate the files on the hub" in lowered
            and "local disk" in lowered
            and "revision" in lowered
        )
    )


def _humanize_translation_error(exc: Exception, model_name: str) -> str:
    message = str(exc)
    lowered = message.lower()
    if "download" in lowered or "network" in lowered or "connection" in lowered:
        return (
            f"Translation model download failed for '{model_name}': {message}. "
            "Check internet access and retry once to populate cache. "
            "You can continue without translation via `--no-translate`."
        )
    if "out of memory" in lowered:
        return (
            f"Translation model '{model_name}' ran out of memory: {message}. "
            "Qwen3.5 4B is still memory-intensive; use a machine with more RAM/VRAM or switch to a smaller model."
        )
    if _is_hf_snapshot_cache_error(exc):
        return (
            f"Translation model cache is incomplete/corrupted for '{model_name}': {message}. "
            "Retry once with internet access to refresh cache. "
            "If it persists, clear local Hugging Face cache and retry."
        )
    return f"Translation failed: {message}"


def clear_translation_cache() -> None:
    _TRANSLATION_CACHE.clear()


def _run_translation_batch(
    tokenizer: object,
    model: object,
    messages_batch: list[list[dict[str, str]]],
    max_new_tokens: int,
) -> list[str]:
    if not messages_batch:
        return []

    prompts = [
        _render_chat_prompt(tokenizer, messages)
        for messages in messages_batch
    ]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded = _move_encoded_inputs(encoded, device=_model_input_device(model))

    try:
        import torch

        context = torch.inference_mode()
    except Exception:
        context = nullcontext()

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    elif eos_token_id is not None:
        generation_kwargs["pad_token_id"] = eos_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    with context:
        generated = model.generate(**encoded, **generation_kwargs)

    attention_mask = encoded.get("attention_mask")
    input_ids = encoded["input_ids"]
    decoded: list[str] = []
    for row_idx in range(len(prompts)):
        prompt_tokens = _prompt_token_count(input_ids, attention_mask, row_idx)
        new_tokens = generated[row_idx, prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded.append(_clean_translation_output(text))

    return decoded


def _render_chat_prompt(tokenizer: object, messages: list[dict[str, str]]) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    prompt_lines = []
    for message in messages:
        prompt_lines.append(f"{message['role'].upper()}: {message['content']}")
    prompt_lines.append("ASSISTANT:")
    return "\n\n".join(prompt_lines)


def _move_encoded_inputs(encoded: object, device: object) -> dict[str, object]:
    if not isinstance(encoded, dict):
        return encoded
    out: dict[str, object] = {}
    for key, value in encoded.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def _model_input_device(model: object) -> object:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _prompt_token_count(input_ids: object, attention_mask: object, row_idx: int) -> int:
    if attention_mask is None:
        return int(input_ids[row_idx].shape[0])
    return int(attention_mask[row_idx].sum().item())


def _clean_translation_output(text: str) -> str:
    cleaned = _THINK_TAG_RE.sub("", text).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) == 1:
        cleaned = lines[0]
    elif lines:
        cleaned = " ".join(lines)
    cleaned = _OUTPUT_PREFIX_RE.sub("", cleaned).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned
