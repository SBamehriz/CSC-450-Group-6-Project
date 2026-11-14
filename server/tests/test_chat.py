import json
import uuid

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny import make_forge_model  # noqa: E402

from forge import chat_api, measuring  # noqa: E402
from forge.chatformat import END_ID, SPECIAL_TOKENS  # noqa: E402

USER_ID = SPECIAL_TOKENS.index("<|user|>")


@pytest.fixture(autouse=True)
def no_card(monkeypatch):
    monkeypatch.setattr(measuring, "card_room", lambda: None)
    chat_api.unload_models()


def ask(client, model, content="tell me about the river", **settings) -> list[dict]:
    body = {"model_id": str(model.id), "messages": [{"role": "user", "content": content}]}
    response = client.post("/api/chat", json=body | settings)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in response.text.split("\n\n") if line]
    assert all(line.startswith("data: ") for line in lines)
    return [json.loads(line.removeprefix("data: ")) for line in lines]


def text_of(events: list[dict]) -> str:
    return "".join(event.get("text", "") for event in events)


def test_an_answer_streams_in_pieces_then_says_it_is_done(client, db):
    model = make_forge_model(db, always=200)
    events = ask(client, model, temperature=0, max_new_tokens=8)
    assert len(events) >= 2 and all("text" in event for event in events[:-1])
    assert events[-1]["done"] is True
    assert events[-1]["tokens"] == 8 and events[-1]["stopped"] == "length"
    assert events[-1]["seconds"] >= 0
    assert text_of(events)


def test_an_answer_stops_at_the_end_mark(client, db):
    events = ask(client, make_forge_model(db, always=END_ID), temperature=0, max_new_tokens=8)
    assert events == [{**events[-1], "done": True, "tokens": 0, "stopped": "end"}]


def test_special_word_pieces_are_never_shown(client, db):
    model = make_forge_model(db, always=USER_ID)
    events = ask(client, model, temperature=0, max_new_tokens=5)
    assert text_of(events) == "" and events[-1]["tokens"] == 5


def test_a_model_with_marks_reads_the_chat_layout(client, db):
    model = make_forge_model(db, always=200)
    ask(client, model, temperature=0, max_new_tokens=2)
    loaded = chat_api._slots[model.id].loaded
    said = measuring.layout(loaded, [{"role": "user", "content": "hi"}])
    assert said == "<|user|>hi<|end|><|assistant|>"
    assert loaded.ends == (0, END_ID)


def test_a_model_without_marks_gets_the_plain_layout(client, db):
    # id 4 is an ordinary word piece in an older vocabulary, so it does not end the answer
    model = make_forge_model(db, always=END_ID, marks=False)
    events = ask(client, model, temperature=0, max_new_tokens=6)
    assert events[-1]["tokens"] == 6 and events[-1]["stopped"] == "length"
    assert text_of(events)
    loaded = chat_api._slots[model.id].loaded
    said = measuring.layout(loaded, [{"role": "user", "content": "hi"}])
    assert said == "User: hi\nAssistant:"
    assert loaded.ends == (0,)


def test_the_same_seed_and_settings_give_the_same_answer(client, db):
    model = make_forge_model(db)
    settings = {"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 16}
    first = text_of(ask(client, model, seed=7, **settings))
    assert text_of(ask(client, model, seed=7, **settings)) == first
    others = {text_of(ask(client, model, seed=seed, **settings)) for seed in range(8, 12)}
    assert others != {first}


def test_a_long_conversation_is_trimmed_to_fit(client, db):
    model = make_forge_model(db, always=200)
    turns = [{"role": "user", "content": "the river " * 5}] * 20
    events = ask(client, model, temperature=0, max_new_tokens=30, messages=turns)
    assert events[-1]["done"] is True and events[-1]["tokens"] == 30
    loaded = chat_api._slots[model.id].loaded
    ids, room = measuring.fit(loaded, turns, 30)
    assert room == 30 and len(ids) + room <= loaded.window
    assert ids[0] == USER_ID and len(ids) < len(turns) * 5


def test_problems_come_back_as_plain_words(client, db, make_model):
    unknown = ask(client, type("M", (), {"id": uuid.uuid4()}))
    assert unknown == [{"error": "This model was deleted."}]
    untrained = ask(client, make_model())
    assert "not been trained" in untrained[0]["error"]
    too_long = ask(client, make_forge_model(db), content="river " * 300)
    assert "Shorten it" in too_long[0]["error"]


def test_two_models_stay_loaded_and_unload_frees_them(client, db):
    first, second, third = (make_forge_model(db, always=200) for _ in range(3))
    for model in (first, second, third):
        ask(client, model, temperature=0, max_new_tokens=2)
    assert list(chat_api._slots) == [second.id, third.id]
    chat_api.unload_models()
    assert not chat_api._slots


def test_while_a_training_run_waits_or_runs_chat_stays_off_the_card(
    client, db, make_job, monkeypatch
):
    seen = []
    real = measuring.load

    def load(model_id, checkpoint_id=None, asked="auto"):
        seen.append(asked)
        return real(model_id, checkpoint_id, asked)

    monkeypatch.setattr(measuring, "load", load)
    model = make_forge_model(db, always=200)
    ask(client, model, temperature=0, max_new_tokens=2)
    chat_api.unload_models()
    job = make_job("train_run", "running")
    ask(client, model, temperature=0, max_new_tokens=2)
    chat_api.unload_models()
    job.status = "completed"
    db.commit()
    ask(client, model, temperature=0, max_new_tokens=2)
    assert seen == ["auto", "cpu", "auto"]


def test_compare_asks_a_chat_model_through_its_chat_layout(db):
    from forge import prompts

    def spied(loaded, seen: list):
        template = loaded.tokenizer.apply_chat_template

        def apply(turns, **kw):
            seen.append(turns[0]["content"])
            return template(turns, **kw)

        loaded.tokenizer.apply_chat_template = apply
        return loaded

    settings = {"max_new_tokens": 4, "temperature": 0, "top_k": 0}
    for chat, asked in ((True, ["hello"]), (False, [])):
        seen = []
        loaded = measuring.load(make_forge_model(db, chat=chat, always=200).id, None, "cpu")
        assert prompts.answer(spied(loaded, seen), "hello", settings, 0)
        assert seen == asked
    # a chat model stops at the end of its turn
    loaded = measuring.load(make_forge_model(db, chat=True, always=END_ID).id, None, "cpu")
    assert prompts.answer(loaded, "hello", settings, 0) == ""


@pytest.mark.parametrize("marks", [True, False])
def test_an_export_carries_the_chat_layout_for_a_model_that_learned_to_chat(db, marks):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from forge.models import Checkpoint, Job, Run
    from forge.train import export

    model = make_forge_model(db, marks=marks, chat=marks)
    run = db.query(Run).filter(Run.model_id == model.id).one()
    checkpoint = db.query(Checkpoint).filter(Checkpoint.run_id == run.id).one()
    job = Job(job_type="model_export", payload={"checkpoint_id": str(checkpoint.id)})
    db.add(job)
    db.commit()
    folder = export.export_dir(checkpoint.id)
    assert export.run(job.id)["folder"]
    tokenizer = AutoTokenizer.from_pretrained(folder)
    theirs = AutoModelForCausalLM.from_pretrained(folder)
    turns = [{"role": "user", "content": "hi"}]
    if marks:
        said = tokenizer.apply_chat_template(turns, add_generation_prompt=True, tokenize=False)
        assert said == "<|user|>hi<|end|><|assistant|>"
        ids = tokenizer(said, add_special_tokens=False)["input_ids"]
        assert ids[0] == USER_ID and ids[-1] == SPECIAL_TOKENS.index("<|assistant|>")
        assert tokenizer.decode([USER_ID, 200], skip_special_tokens=True) == tokenizer.decode([200])
        assert theirs.generation_config.eos_token_id == [0, END_ID]
    else:
        assert not tokenizer.chat_template
        assert theirs.generation_config.eos_token_id == 0


def test_a_model_too_big_for_the_free_card_runs_on_the_processor(client, db, monkeypatch):
    seen = []
    real = measuring.load

    def load(model_id, checkpoint_id=None, asked="auto"):
        seen.append(asked)
        return real(model_id, checkpoint_id, "cpu")

    monkeypatch.setattr(measuring, "load", load)
    monkeypatch.setattr(measuring, "card_room", lambda: 1000)
    ask(client, make_forge_model(db, always=200), temperature=0, max_new_tokens=2)
    monkeypatch.setattr(measuring, "card_room", lambda: 10**9)
    ask(client, make_forge_model(db, always=200), temperature=0, max_new_tokens=2)
    assert seen == ["cpu", "auto"]
