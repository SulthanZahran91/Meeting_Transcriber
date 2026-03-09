from meeting_transcriber.postprocess import postprocess_segments
from meeting_transcriber.transcribe import Segment


def _seg(start, end, text):
    return Segment(start=start, end=end, text=text)


class TestCleanDisfluencies:
    def test_removes_filler_words(self):
        segments = [_seg(0, 2, "음 그래서 회의를 시작하겠습니다")]
        result = postprocess_segments(segments)
        assert len(result) == 1
        assert "음" not in result[0].text
        assert "회의를" in result[0].text

    def test_removes_trailing_dots(self):
        segments = [_seg(0, 1, "네 알겠습니다.....")]
        result = postprocess_segments(segments)
        assert not result[0].text.endswith(".....")

    def test_empty_after_cleaning_is_dropped(self):
        segments = [_seg(0, 1, "음... 어...")]
        result = postprocess_segments(segments)
        assert len(result) == 0

    def test_preserves_meaningful_text(self):
        segments = [_seg(0, 3, "이번 분기 매출이 증가했습니다")]
        result = postprocess_segments(segments)
        assert result[0].text == "이번 분기 매출이 증가했습니다"


class TestMergeShortSegments:
    def test_merges_adjacent_short_segments(self):
        segments = [
            _seg(0.0, 0.5, "네"),
            _seg(0.6, 1.5, "알겠습니다"),
        ]
        result = postprocess_segments(segments, min_merge_chars=8)
        assert len(result) == 1
        assert "네" in result[0].text
        assert "알겠습니다" in result[0].text
        assert result[0].start == 0.0
        assert result[0].end == 1.5

    def test_does_not_merge_with_large_gap(self):
        segments = [
            _seg(0.0, 0.5, "네"),
            _seg(5.0, 6.0, "다음 안건"),
        ]
        result = postprocess_segments(segments, min_merge_chars=8, max_merge_gap=1.5)
        assert len(result) == 2

    def test_does_not_merge_long_segments(self):
        segments = [
            _seg(0, 3, "이번 분기 매출이 크게 증가했습니다"),
            _seg(3.1, 6, "다음 분기 계획을 논의하겠습니다"),
        ]
        result = postprocess_segments(segments, min_merge_chars=8)
        assert len(result) == 2


class TestSplitLongSegments:
    def test_splits_at_punctuation(self):
        long_text = "첫 번째 문장입니다. 두 번째 문장입니다."
        segments = [_seg(0, 10, long_text)]
        result = postprocess_segments(segments, max_segment_length=15)
        assert len(result) == 2
        assert result[0].text == "첫 번째 문장입니다."
        assert result[1].text == "두 번째 문장입니다."

    def test_preserves_short_segments(self):
        segments = [_seg(0, 5, "짧은 문장")]
        result = postprocess_segments(segments, max_segment_length=500)
        assert len(result) == 1

    def test_split_timestamps_proportional(self):
        long_text = "가나다라마바사. 아자차카타파하."
        segments = [_seg(0, 10, long_text)]
        result = postprocess_segments(segments, max_segment_length=10)
        assert len(result) == 2
        assert result[0].start == 0
        assert result[1].end == 10
        assert result[0].end == result[1].start


class TestEndToEnd:
    def test_clean_merge_split_pipeline(self):
        segments = [
            _seg(0.0, 0.3, "음"),
            _seg(0.4, 0.8, "네"),
            _seg(1.0, 5.0, "이번 분기 매출이 증가했습니다"),
        ]
        result = postprocess_segments(segments, min_merge_chars=8)
        # "음" is cleaned away, "네" is short and may merge or stand alone
        # main content is preserved
        assert any("매출" in seg.text for seg in result)

    def test_empty_input(self):
        assert postprocess_segments([]) == []
