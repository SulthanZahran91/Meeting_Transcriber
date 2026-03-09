from __future__ import annotations

import re

from meeting_transcriber.transcribe import Segment

# Korean filler words / disfluencies to strip.
_FILLER_PATTERN = re.compile(
    r"(?:^|\s*)"
    r"(?:음+|어+|아+|그+|에+|야|인제|뭐|저기|있잖아|그러니까|근데|그래서)"
    r"[.…·~]*"
    r"(?:\s+|$)",
)

# Collapse whitespace left after filler removal.
_MULTI_SPACE = re.compile(r"\s{2,}")

# Repeated trailing ellipses / dots.
_TRAILING_DOTS = re.compile(r"[.…]{3,}$")

# Default thresholds.
MIN_MERGE_CHARS = 8
MAX_MERGE_GAP_SECONDS = 1.5


def postprocess_segments(
    segments: list[Segment],
    max_segment_length: int = 500,
    min_merge_chars: int = MIN_MERGE_CHARS,
    max_merge_gap: float = MAX_MERGE_GAP_SECONDS,
) -> list[Segment]:
    cleaned = [_clean_disfluencies(seg) for seg in segments]
    cleaned = [seg for seg in cleaned if seg.text]
    merged = _merge_short_segments(cleaned, min_merge_chars, max_merge_gap)
    split = _split_long_segments(merged, max_segment_length)
    return split


def _clean_disfluencies(seg: Segment) -> Segment:
    text = seg.text
    text = _FILLER_PATTERN.sub(" ", text)
    text = _TRAILING_DOTS.sub("", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return Segment(start=seg.start, end=seg.end, text=text)


def _merge_short_segments(
    segments: list[Segment],
    min_chars: int,
    max_gap: float,
) -> list[Segment]:
    if not segments:
        return []

    result: list[Segment] = []
    buf = segments[0]

    for seg in segments[1:]:
        gap = seg.start - buf.end
        buf_short = len(buf.text) < min_chars
        seg_short = len(seg.text) < min_chars

        if (buf_short or seg_short) and gap <= max_gap:
            buf = Segment(
                start=buf.start,
                end=seg.end,
                text=f"{buf.text} {seg.text}",
            )
        else:
            result.append(buf)
            buf = seg

    result.append(buf)
    return result


def _split_long_segments(segments: list[Segment], max_length: int) -> list[Segment]:
    if max_length <= 0:
        return segments

    result: list[Segment] = []
    for seg in segments:
        if len(seg.text) <= max_length:
            result.append(seg)
            continue
        parts = _split_text(seg.text, max_length)
        duration = seg.end - seg.start
        total_chars = sum(len(p) for p in parts)
        offset = seg.start
        for part in parts:
            frac = len(part) / total_chars if total_chars > 0 else 1.0 / len(parts)
            part_dur = duration * frac
            result.append(Segment(start=offset, end=offset + part_dur, text=part))
            offset += part_dur
    return result


_SPLIT_POINTS = re.compile(r"(?<=[.?!。])\s+|(?<=[,，、;])\s+")


def _split_text(text: str, max_length: int) -> list[str]:
    if len(text) <= max_length:
        return [text]

    candidates = list(_SPLIT_POINTS.finditer(text))
    if candidates:
        mid = len(text) // 2
        best = min(candidates, key=lambda m: abs(m.start() - mid))
        left = text[: best.start()].strip()
        right = text[best.end() :].strip()
        if left and right:
            return _split_text(left, max_length) + _split_text(right, max_length)

    # No punctuation split point — hard split at max_length boundary on a space.
    pos = text.rfind(" ", 0, max_length)
    if pos <= 0:
        pos = max_length
    left = text[:pos].strip()
    right = text[pos:].strip()
    if not right:
        return [left]
    return [left] + _split_text(right, max_length)
