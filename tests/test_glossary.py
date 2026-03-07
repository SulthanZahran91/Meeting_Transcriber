from pathlib import Path

from meeting_transcriber.glossary import apply_glossary
from meeting_transcriber.translate import TranslatedSegment

FIXTURE_GLOSSARY = (
    Path(__file__).parent / "fixtures" / "samples" / "glossary.sample.json"
)


def test_apply_glossary_corrections_and_overrides(tmp_path: Path) -> None:
    glossary_path = tmp_path / "glossary.json"
    glossary_path.write_text(
        FIXTURE_GLOSSARY.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    segments = [
        TranslatedSegment(0.0, 1.0, "라인 반송 시스템", "conveyer uses stoker"),
        TranslatedSegment(1.0, 2.0, "일반 문장", "normal english"),
    ]

    out = apply_glossary(segments, glossary_path)

    assert out[0].english == "conveyor uses stocker (transport/conveyance)"
    assert out[1].english == "normal english"
    assert segments[0].english == "conveyer uses stoker"


def test_apply_glossary_missing_file_returns_copy() -> None:
    segments = [TranslatedSegment(0.0, 1.0, "문장", "sentence")]
    out = apply_glossary(segments, Path("missing-glossary.json"))
    assert out == segments
    assert out is not segments


def test_apply_glossary_invalid_json_errors(tmp_path: Path) -> None:
    glossary_path = tmp_path / "bad.json"
    glossary_path.write_text("{bad json", encoding="utf-8")

    try:
        apply_glossary([TranslatedSegment(0.0, 1.0, "문장", "sentence")], glossary_path)
    except RuntimeError as exc:
        assert "Invalid glossary JSON" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for malformed glossary JSON")
