from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

from meeting_transcriber.translate import TranslatedSegment


def apply_glossary(
    segments: list[TranslatedSegment], glossary_path: Path
) -> list[TranslatedSegment]:
    if not glossary_path.exists():
        return segments[:]

    data = _load_glossary(glossary_path)
    corrections = data.get("corrections", {})
    korean_overrides = data.get("korean_overrides", {})

    correction_rules = _compile_corrections(corrections)
    transformed: list[TranslatedSegment] = []

    for seg in segments:
        english = seg.english
        for pattern, replacement in correction_rules:
            english = pattern.sub(replacement, english)
        for korean_term, override in korean_overrides.items():
            if korean_term and korean_term in seg.korean and override:
                if not english:
                    english = override
                elif override.lower() not in english.lower():
                    english = f"{english} ({override})"
        transformed.append(replace(seg, english=english))

    return transformed


def _load_glossary(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid glossary JSON: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Glossary file must contain a JSON object: {path}")
    return payload


def _compile_corrections(corrections: dict) -> list[tuple[re.Pattern[str], str]]:
    rules: list[tuple[re.Pattern[str], str]] = []
    for source, target in corrections.items():
        if not source:
            continue
        pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
        rules.append((pattern, str(target)))
    return rules

