import json

import pytest

pytest.importorskip("torch")

from tiny import make_forge_model  # noqa: E402

from forge import hub  # noqa: E402


def test_a_model_knows_its_family_and_whether_it_is_ready(client, db, make_model):
    base = make_forge_model(db, name="base")
    child = make_forge_model(db, name="child", chat=True)
    child.parent_id = base.id
    blank = make_model(name="blank")
    db.commit()

    got = client.get(f"/api/models/{child.id}").json()
    assert got["parent_id"] == str(base.id) and got["parent_name"] == "base"
    assert got["chat"] is True and got["ready"] is True and got["best_val_loss"] == 2.0
    empty = client.get(f"/api/models/{blank.id}").json()
    assert empty["ready"] is False and empty["best_val_loss"] is None
    assert empty["parent_name"] is None and empty["chat"] is False

    ready = client.get("/api/models?ready=true").json()
    assert {one["name"] for one in ready["items"]} == {"base", "child"} and ready["total"] == 2
    waiting = client.get("/api/models?ready=false").json()
    assert [one["name"] for one in waiting["items"]] == ["blank"]
    assert client.get("/api/models").json()["total"] == 3


def test_deleting_a_base_model_keeps_its_fine_tunes(client, db, make_model):
    base = make_model(name="base")
    child = make_model(name="child")
    child.parent_id = base.id
    db.commit()
    assert client.delete(f"/api/models/{base.id}").status_code == 204
    got = client.get(f"/api/models/{child.id}").json()
    assert got["parent_id"] is None and got["parent_name"] is None


def test_a_hub_model_is_ready_when_its_files_are_here(client, db, data_dir):
    from forge.models import Model

    folder = data_dir / "models" / "here"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        Model(name=name, n_layer=1, d_model=8, n_head=1, ctx_len=8, source="hub", artifact_uri=uri)
        for name, uri in (("here", "models/here"), ("gone", "models/gone"))
    ]
    db.add_all(rows)
    db.commit()
    assert [client.get(f"/api/models/{row.id}").json()["ready"] for row in rows] == [True, False]


def test_the_size_presets_are_gone(client):
    assert client.get("/api/models/presets").status_code != 200


def test_a_hub_import_knows_a_chat_template(tmp_path):
    assert not hub.has_chat_template(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "x"}))
    assert not hub.has_chat_template(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ x }}"}))
    assert hub.has_chat_template(tmp_path)
    (tmp_path / "tokenizer_config.json").unlink()
    (tmp_path / "chat_template.jinja").write_text("{{ x }}")
    assert hub.has_chat_template(tmp_path)
