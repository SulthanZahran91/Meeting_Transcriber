from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Optional

from meeting_transcriber.translate import TranslatedSegment


def format_output(
    segments: list[TranslatedSegment],
    format: str,
    output_path: Path,
    metadata: Optional[dict[str, str]] = None,
) -> Path:
    format = format.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}

    if format == "html":
        content = _render_html(segments, metadata)
        output_path.write_text(content, encoding="utf-8")
        return output_path
    if format == "markdown":
        content = _render_markdown(segments)
        output_path.write_text(content, encoding="utf-8")
        return output_path
    if format == "srt":
        english_srt = _render_srt(segments, korean=False)
        korean_srt = _render_srt(segments, korean=True)
        output_path.write_text(english_srt, encoding="utf-8")
        korean_path = output_path.with_name(f"{output_path.stem}.ko.srt")
        korean_path.write_text(korean_srt, encoding="utf-8")
        return output_path
    raise ValueError(f"Unsupported output format: {format}")


def format_timestamp(seconds: float, srt: bool = False) -> str:
    millis = max(0, int(round(seconds * 1000)))
    hh, rem = divmod(millis, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    separator = "," if srt else "."
    return f"{hh:02}:{mm:02}:{ss:02}{separator}{ms:03}"


def _render_html(segments: list[TranslatedSegment], metadata: dict[str, str]) -> str:
    source = metadata.get("source", "unknown")
    model = metadata.get("model", "unknown")
    processing_time = metadata.get("processing_time", "unknown")
    run_date = metadata.get("date", datetime.now(timezone.utc).isoformat())

    rows = []
    for seg in segments:
        time_txt = format_timestamp(seg.start)
        left = f"<span class='ts'>{time_txt}</span><div>{escape(seg.korean)}</div>"
        right = escape(seg.english)
        rows.append(f"<tr><td>{left}</td><td>{right}</td></tr>")
    joined_rows = "\n".join(rows)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Meeting Transcript</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f8fafc;
    }}
    h1 {{ margin: 0 0 8px; }}
    .meta {{
      margin-bottom: 16px;
      color: #334155;
      line-height: 1.6;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: #ffffff;
      border: 1px solid #cbd5e1;
    }}
    td {{
      border-bottom: 1px solid #e2e8f0;
      vertical-align: top;
      padding: 10px 12px;
      width: 50%;
    }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    .ts {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.9rem;
      color: #475569;
      display: inline-block;
      margin-bottom: 6px;
    }}
  </style>
</head>
<body>
  <h1>Meeting Transcript</h1>
  <div class="meta">
    <div><strong>Source:</strong> {escape(source)}</div>
    <div><strong>Date:</strong> {escape(run_date)}</div>
    <div><strong>Model:</strong> {escape(model)}</div>
    <div><strong>Processing Time:</strong> {escape(processing_time)}</div>
  </div>
  <table>
    <tbody>
      {joined_rows}
    </tbody>
  </table>
</body>
</html>
"""


def _render_markdown(segments: list[TranslatedSegment]) -> str:
    lines = ["| Time | Korean | English |", "|---|---|---|"]
    for seg in segments:
        time_txt = format_timestamp(seg.start)
        korean = _escape_markdown(seg.korean)
        english = _escape_markdown(seg.english)
        lines.append(f"| {time_txt} | {korean} | {english} |")
    return "\n".join(lines) + "\n"


def _render_srt(segments: list[TranslatedSegment], korean: bool) -> str:
    blocks = []
    for idx, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.start, srt=True)
        end = format_timestamp(seg.end, srt=True)
        text = seg.korean if korean else seg.english
        blocks.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def _escape_markdown(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")
