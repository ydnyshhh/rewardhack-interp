from __future__ import annotations

from rewardhack_interp.config import ModelConfig
from rewardhack_interp.modeling import render_prompt_with_tokenizer


class TokenizerWithThinkingSwitch:
    chat_template = "qwen"

    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] | None = None
        self.last_enable_thinking: bool | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.last_messages = messages
        self.last_enable_thinking = enable_thinking
        return "rendered"


class TokenizerWithoutThinkingSwitch:
    chat_template = "qwen"

    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.last_messages = messages
        return "rendered"


def test_render_prompt_passes_enable_thinking_false_when_supported() -> None:
    tokenizer = TokenizerWithThinkingSwitch()
    config = ModelConfig(
        model_name_or_path="Qwen/Qwen3-4B",
        enable_thinking=False,
        system_prompt="Return only code.",
    )

    rendered = render_prompt_with_tokenizer(
        tokenizer=tokenizer,
        config=config,
        user_prompt="Write the function.",
    )

    assert rendered == "rendered"
    assert tokenizer.last_enable_thinking is False
    assert tokenizer.last_messages == [
        {"role": "system", "content": "Return only code."},
        {"role": "user", "content": "Write the function."},
    ]


def test_render_prompt_omits_enable_thinking_when_unsupported() -> None:
    tokenizer = TokenizerWithoutThinkingSwitch()
    config = ModelConfig(
        model_name_or_path="Qwen/Qwen3-4B",
        enable_thinking=False,
        system_prompt="Return only code.",
    )

    rendered = render_prompt_with_tokenizer(
        tokenizer=tokenizer,
        config=config,
        user_prompt="Write the function.",
    )

    assert rendered == "rendered"
    assert tokenizer.last_messages == [
        {"role": "system", "content": "Return only code."},
        {"role": "user", "content": "Write the function."},
    ]
