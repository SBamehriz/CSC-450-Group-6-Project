import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select

from forge.models import Model, Run
from forge.train.presets import (
    MAX_D_MODEL,
    MAX_DROPOUT,
    MAX_N_LAYER,
    ModelConfig,
    count_params,
    explain_shape_error,
    training_memory,
    validate,
)


def test_model_shape_validation_rules():
    # Valid config
    cfg = ModelConfig(n_layer=4, d_model=128, n_head=4, ctx_len=512, vocab_size=4096)
    validate(cfg)

    # Invalid head divisibility
    err = explain_shape_error(n_layer=4, d_model=130, n_head=4, ctx_len=512)
    assert err is not None
    assert "must divide evenly by attention heads" in err

    # Invalid layer count
    err = explain_shape_error(n_layer=0, d_model=128, n_head=4, ctx_len=512)
    assert err is not None
    assert f"between 1 and {MAX_N_LAYER}" in err

    err = explain_shape_error(n_layer=MAX_N_LAYER + 1, d_model=128, n_head=4, ctx_len=512)
    assert err is not None
    assert f"between 1 and {MAX_N_LAYER}" in err

    # Invalid width
    err = explain_shape_error(n_layer=4, d_model=MAX_D_MODEL + 4, n_head=4, ctx_len=512)
    assert err is not None
    assert f"between 1 and {MAX_D_MODEL}" in err

    # Invalid context length
    err = explain_shape_error(n_layer=4, d_model=128, n_head=4, ctx_len=100)
    assert err is not None
    assert "Context length must be" in err

    # Invalid dropout
    err = explain_shape_error(n_layer=4, d_model=128, n_head=4, ctx_len=512, dropout=0.5)
    assert err is not None
    assert f"Dropout must be between 0.0 and {MAX_DROPOUT}" in err


def test_preview_calculations_and_training_memory():
    cfg = ModelConfig(n_layer=4, d_model=128, n_head=4, ctx_len=512, vocab_size=4096)
    params = count_params(cfg)
    assert params > 0
    # training_memory provides estimated memory parts dict
    mem = training_memory(cfg, micro_batch_seqs=8)
    total_bytes = sum(mem.values())
    assert total_bytes > params * 16  # weights + grads + Adam + activations


def test_create_model(client: TestClient, db):
    name = f"test-nano-{uuid.uuid4().hex[:6]}"
    resp = client.post(
        "/api/models",
        json={
            "name": name,
            "description": "Nano test",
            "n_layer": 4,
            "d_model": 128,
            "n_head": 4,
            "ctx_len": 512,
            "vocab_size": 16384,
            "dropout": 0.0,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == name
    assert data["param_count"] > 0
    assert data["estimated_memory_bytes"] is not None

    # Verify saved in database
    saved = db.execute(select(Model).where(Model.name == name)).scalar_one()
    assert saved.n_layer == 4
    assert saved.d_model == 128


def test_model_duplicate_name_rejected(client: TestClient):
    name = f"dupe-model-{uuid.uuid4().hex[:6]}"
    resp1 = client.post(
        "/api/models",
        json={
            "name": name,
            "n_layer": 4,
            "d_model": 128,
            "n_head": 4,
            "ctx_len": 512,
        },
    )
    assert resp1.status_code == 201

    resp2 = client.post(
        "/api/models",
        json={
            "name": name,
            "n_layer": 4,
            "d_model": 128,
            "n_head": 4,
            "ctx_len": 512,
        },
    )
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["error"]["message"]


def test_model_deletion_protection(client: TestClient, db, make_model):
    model = make_model(name=f"model-del-{uuid.uuid4().hex[:6]}")
    run = Run(
        name=f"run-{uuid.uuid4().hex[:6]}",
        model_id=model.id,
        status="completed",
    )
    db.add(run)
    db.commit()

    # Attempt delete while run references it
    del_resp = client.delete(f"/api/models/{model.id}")
    assert del_resp.status_code == 409
    assert "is used by 1 training run" in del_resp.json()["error"]["message"]

    # Delete referencing run, now deletion succeeds
    db.delete(run)
    db.commit()
    del_resp2 = client.delete(f"/api/models/{model.id}")
    assert del_resp2.status_code == 204
