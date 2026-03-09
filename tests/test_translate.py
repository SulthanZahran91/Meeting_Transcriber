from __future__ import annotations

from pathlib import Path
import sys
import types

from meeting_transcriber.config import TranscriberConfig
from meeting_transcriber.transcribe import Segment
from meeting_transcriber import translate
from meeting_transcriber.translate import translate_segments
from rich.console import Console


def test_build_translation_messages_uses_three_segments_of_context() -> None:
    segments = [
        Segment(0.0, 1.0, "영"),
        Segment(1.0, 2.0, "일"),
        Segment(2.0, 3.0, "이"),
        Segment(3.0, 4.0, "삼"),
        Segment(4.0, 5.0, "사"),
        Segment(5.0, 6.0, "오"),
        Segment(6.0, 7.0, "육"),
        Segment(7.0, 8.0, "칠"),
        Segment(8.0, 9.0, "팔"),
    ]

    messages = translate._build_translation_messages(segments, target_index=4, context_window=3)
    user_prompt = messages[1]["content"]

    assert "CTX-3: 일" in user_prompt
    assert "CTX-2: 이" in user_prompt
    assert "CTX-1: 삼" in user_prompt
    assert "TARGET: 사" in user_prompt
    assert "CTX+1: 오" in user_prompt
    assert "CTX+2: 육" in user_prompt
    assert "CTX+3: 칠" in user_prompt
    assert "영" not in user_prompt
    assert "팔" not in user_prompt


def test_translate_batches_and_keeps_alignment(monkeypatch) -> None:
    batch_calls: list[list[str]] = []
    monkeypatch.setattr(
        "meeting_transcriber.translate._load_translation_model",
        lambda *_args, **_kwargs: (object(), object()),
    )

    def fake_run(
        _tokenizer: object,
        _model: object,
        messages_batch: list[list[dict[str, str]]],
        max_new_tokens: int,
    ) -> list[str]:
        assert max_new_tokens == TranscriberConfig().translation_max_new_tokens
        targets = [
            next(
                line.split(": ", 1)[1]
                for line in messages[1]["content"].splitlines()
                if line.startswith("TARGET: ")
            )
            for messages in messages_batch
        ]
        batch_calls.append(targets[:])
        return [f"EN:{txt}" for txt in targets]

    monkeypatch.setattr("meeting_transcriber.translate._run_translation_batch", fake_run)

    segments = [
        Segment(0.0, 1.0, "하나"),
        Segment(1.0, 2.0, ""),
        Segment(2.0, 3.0, "둘"),
        Segment(3.0, 4.0, "셋"),
        Segment(4.0, 5.0, "넷"),
    ]
    out = translate_segments(
        segments,
        TranscriberConfig(),
        batch_size=2,
    )

    assert len(out) == 5
    assert batch_calls == [["하나", "둘"], ["셋", "넷"]]
    assert out[1].english == ""
    assert out[2].english == "EN:둘"
    assert out[3].start == 3.0
    assert out[3].end == 4.0


def test_translate_skips_whitespace_segments(monkeypatch) -> None:
    monkeypatch.setattr(
        "meeting_transcriber.translate._load_translation_model",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        "meeting_transcriber.translate._run_translation_batch",
        lambda _tokenizer, _model, messages_batch, max_new_tokens: [
            "EN:"
            + next(
                line.split(": ", 1)[1]
                for line in messages[1]["content"].splitlines()
                if line.startswith("TARGET: ")
            )
            for messages in messages_batch
        ],
    )

    segments = [
        Segment(0.0, 1.0, "  "),
        Segment(1.0, 2.0, "\n"),
        Segment(2.0, 3.0, "정상"),
    ]
    out = translate_segments(segments, TranscriberConfig(), batch_size=1)

    assert [item.english for item in out] == ["", "", "EN:정상"]


def test_clean_translation_output_strips_labels_and_quotes() -> None:
    assert translate._clean_translation_output('English: "Hello there"') == "Hello there"
    assert translate._clean_translation_output("Translation:\nMeeting starts") == "Meeting starts"
    assert translate._clean_translation_output("<think>reasoning</think>\nHello") == "Hello"


def test_render_chat_prompt_disables_thinking_when_supported() -> None:
    captured: dict[str, object] = {}

    class _FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "prompt"

    prompt = translate._render_chat_prompt(
        _FakeTokenizer(),
        [{"role": "user", "content": "TARGET: 안녕하세요"}],
    )

    assert prompt == "prompt"
    assert captured["kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_load_translation_model_retries_snapshot_cache_error(monkeypatch, tmp_path: Path) -> None:
    snapshot_calls: list[dict[str, object]] = []
    model_dir = tmp_path / "translation-model"

    class _FakeTokenizer:
        calls = 0

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            cls.calls += 1
            if cls.calls == 1:
                raise RuntimeError(
                    "An error happened while trying to locate the files on the Hub and "
                    "we cannot find the appropriate snapshot folder for the specified "
                    "revision on the local disk."
                )
            assert args[0] == str(model_dir)
            assert kwargs.get("local_files_only") is True
            assert kwargs.get("trust_remote_code") is True
            return types.SimpleNamespace(
                pad_token=None,
                pad_token_id=None,
                eos_token="<eos>",
                eos_token_id=7,
            )

    class _FakeLoadedModel:
        def eval(self) -> None:
            return None

    class _FakeModel:
        calls = 0

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> _FakeLoadedModel:
            cls.calls += 1
            assert args[0] == str(model_dir)
            assert kwargs.get("local_files_only") is True
            assert kwargs.get("trust_remote_code") is True
            return _FakeLoadedModel()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        AutoModelForCausalLM=_FakeModel,
    )
    fake_hf_hub = types.SimpleNamespace(
        snapshot_download=lambda **kwargs: snapshot_calls.append(kwargs)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    monkeypatch.setattr(
        "meeting_transcriber.translate._translation_model_dir",
        lambda _model_name: model_dir,
    )
    monkeypatch.setattr(
        "meeting_transcriber.translate._translation_model_load_kwargs",
        lambda: {"trust_remote_code": True},
    )
    translate.clear_translation_cache()

    translate._load_translation_model(  # type: ignore[attr-defined]
        TranscriberConfig(translation_model="demo/model"),
        console=Console(record=True),
    )

    assert _FakeTokenizer.calls == 2
    assert _FakeModel.calls == 1
    assert snapshot_calls == [
        {
            "repo_id": "demo/model",
            "local_dir": str(model_dir),
            "force_download": True,
            "local_files_only": False,
        }
    ]


def test_load_translation_model_uses_local_cache_when_hub_blocked(
    monkeypatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "translation-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    class _FakeTokenizer:
        calls = 0

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            cls.calls += 1
            if cls.calls == 1:
                raise RuntimeError(
                    "An error happened while trying to locate the files on the Hub and "
                    "we cannot find the appropriate snapshot folder for the specified "
                    "revision on the local disk."
                )
            assert args[0] == str(model_dir)
            assert kwargs.get("local_files_only") is True
            assert kwargs.get("trust_remote_code") is True
            return types.SimpleNamespace(
                pad_token=None,
                pad_token_id=None,
                eos_token="<eos>",
                eos_token_id=7,
            )

    class _FakeLoadedModel:
        def eval(self) -> None:
            return None

    class _FakeModel:
        calls = 0

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> _FakeLoadedModel:
            cls.calls += 1
            assert args[0] == str(model_dir)
            assert kwargs.get("local_files_only") is True
            assert kwargs.get("trust_remote_code") is True
            return _FakeLoadedModel()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        AutoModelForCausalLM=_FakeModel,
    )
    fake_hf_hub = types.SimpleNamespace(
        snapshot_download=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot_download should not be called")
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    monkeypatch.setattr(
        "meeting_transcriber.translate._translation_model_dir",
        lambda _model_name: model_dir,
    )
    monkeypatch.setattr(
        "meeting_transcriber.translate._translation_model_load_kwargs",
        lambda: {"trust_remote_code": True},
    )
    translate.clear_translation_cache()

    translate._load_translation_model(  # type: ignore[attr-defined]
        TranscriberConfig(translation_model="demo/model"),
        console=Console(record=True),
    )

    assert _FakeTokenizer.calls == 2
    assert _FakeModel.calls == 1
