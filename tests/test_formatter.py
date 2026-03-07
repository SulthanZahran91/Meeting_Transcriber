from pathlib import Path

from meeting_transcriber.formatter import format_output, format_timestamp
from meeting_transcriber.translate import TranslatedSegment


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

