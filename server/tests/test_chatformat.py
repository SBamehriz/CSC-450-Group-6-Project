import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from forge import chatformat

TALK = [
    {"role": "system", "content": "Be kind."},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hello there"},
]


def test_a_conversation_is_plain_text_with_marks():
    text = chatformat.render(TALK)
    assert text == "<|system|>Be kind.<|end|><|user|>hello<|end|><|assistant|>hello there<|end|>"
    assert chatformat.render(TALK[:2], add_generation_prompt=True).endswith("<|assistant|>")


def test_only_the_assistant_words_and_their_end_mark_are_learned():
    # system 9 end user 8 end assistant 7 6 end eos user 5 end assistant 4 end
    ids = np.array([1, 9, 4, 2, 8, 4, 3, 7, 6, 4, 0, 2, 5, 4, 3, 4, 4])
    learn = chatformat.learned(ids).astype(int).tolist()
    assert learn == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]


def test_learned_works_on_a_batch_and_on_text_with_no_marks():
    ids = np.array([[3, 7, 4, 9], [9, 9, 9, 9]])
    assert chatformat.learned(ids).astype(int).tolist() == [[0, 1, 1, 0], [0, 0, 0, 0]]


@pytest.mark.parametrize(
    "messages",
    [
        None,
        [],
        [{"role": "user", "content": "hi"}],
        [{"role": "robot", "content": "hi"}, {"role": "assistant", "content": "x"}],
        [{"role": "user", "content": 3}, {"role": "assistant", "content": "x"}],
        [{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}],
    ],
)
def test_bad_conversations_are_named(messages):
    assert chatformat.problem(messages)


def test_a_good_conversation_has_no_problem():
    assert chatformat.problem(TALK) is None


def test_the_app_starts_without_the_training_extras():
    # a basic install has no numpy or torch
    code = (
        "import sys; sys.modules['numpy'] = sys.modules['torch'] = None; "
        "from forge.main import create_app; create_app().openapi()"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
