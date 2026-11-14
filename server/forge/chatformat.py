# how a conversation becomes text; every new vocabulary reserves these marks

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

EOS = "<|eos|>"
END = "<|end|>"
ROLES = ("system", "user", "assistant")
# order matters: the position is the id
SPECIAL_TOKENS = (EOS, "<|system|>", "<|user|>", "<|assistant|>", END)
EOS_ID = 0
ASSISTANT_ID = 3
END_ID = 4

# the same layout for transformers, so chat and export agree with training
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def problem(messages: object) -> str | None:
    if not isinstance(messages, list) or not messages:
        return "it is not a list of turns"
    for index, turn in enumerate(messages):
        if not isinstance(turn, dict):
            return "a turn is not a role with content"
        role, content = turn.get("role"), turn.get("content")
        if role not in ROLES:
            return f"the role {role!r} is not system, user or assistant"
        if not isinstance(content, str):
            return "a turn has no text"
        if role == "system" and index:
            return "a system turn comes after the first turn"
    if not any(turn["role"] == "assistant" and turn["content"].strip() for turn in messages):
        return "it has no assistant answer"
    return None


def render(messages: list[dict], add_generation_prompt: bool = False) -> str:
    # stored like this, then cut into tokens like any text
    text = "".join(f"<|{turn['role']}|>{turn['content']}{END}" for turn in messages)
    return text + "<|assistant|>" if add_generation_prompt else text


def learned(ids: np.ndarray) -> np.ndarray:
    # only the assistant's words and the end mark after them get learned
    # numpy comes with the train extra, so the app must start without it
    import numpy as np

    ids = np.asarray(ids)
    places = np.arange(ids.shape[-1])
    marks = ids <= END_ID
    last_mark = np.maximum.accumulate(np.where(marks, places, -1), axis=-1)
    last_assistant = np.maximum.accumulate(np.where(ids == ASSISTANT_ID, places, -1), axis=-1)
    # what was the newest mark before this token?
    before_mark = np.concatenate([np.full_like(last_mark[..., :1], -1), last_mark[..., :-1]], -1)
    before_assistant = np.concatenate(
        [np.full_like(last_assistant[..., :1], -1), last_assistant[..., :-1]], -1
    )
    return (before_assistant >= 0) & (before_assistant == before_mark)
