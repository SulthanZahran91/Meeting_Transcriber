from pathlib import Path

from meeting_transcriber.formatter import format_output, format_timestamp
from meeting_transcriber.translate import TranslatedSegment


FIXTURES = Path(__file__).parent / "fixtures" / "expected"


def test_format_timestamp() -> None:
    assert format_timestamp(0.0) == "00:00:00.000"
    assert format_timestamp(1.234) == "00:00:01.234"
    assert format_timestamp(3661.007, srt=True) == "01:01:01,007"


def test_format_output_srt_writes_english_and_korean(tmp_path: Path) -> None:
    segments = [
        TranslatedSegment(0.0, 1.2, "안녕하세요", "Hello"),
        TranslatedSegment(1.2, 2.4, "회의 시작", "Meeting starts"),
    ]
    out_path = tmp_path / "meeting.srt"

    written = format_output(segments, "srt", out_path)
    ko_path = tmp_path / "meeting.ko.srt"

    assert written == out_path
    assert out_path.exists()
    assert ko_path.exists()
    assert "Hello" in out_path.read_text(encoding="utf-8")
    assert "안녕하세요" in ko_path.read_text(encoding="utf-8")


def test_format_output_matches_snapshots(tmp_path: Path) -> None:
    segments = [
        TranslatedSegment(0.0, 1.2, "대한민국", "South Korea"),
        TranslatedSegment(1.2, 2.0, "회의 시작", "Meeting starts"),
    ]
    metadata = {
        "source": "korean_sample.ogg",
        "date": "2026-03-07T00:00:00+00:00",
        "model": "tiny (cpu/int8)",
        "processing_time": "0.1s",
    }

    html_out = tmp_path / "sample.html"
    md_out = tmp_path / "sample.md"
    srt_out = tmp_path / "sample.srt"
    format_output(segments, "html", html_out, metadata=metadata)
    format_output(segments, "markdown", md_out, metadata=metadata)
    format_output(segments, "srt", srt_out, metadata=metadata)

    assert html_out.read_text(encoding="utf-8") == (
        FIXTURES / "sample_output.html"
    ).read_text(encoding="utf-8")
    assert md_out.read_text(encoding="utf-8") == (
        FIXTURES / "sample_output.md"
    ).read_text(encoding="utf-8")
    assert srt_out.read_text(encoding="utf-8") == (
        FIXTURES / "sample_output.srt"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "sample.ko.srt").read_text(encoding="utf-8") == (
        FIXTURES / "sample_output.ko.srt"
    ).read_text(encoding="utf-8")
