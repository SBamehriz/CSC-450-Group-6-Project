"""SP9.8 model configuration import API tests."""

import json

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from forge.db import get_session
from forge.errors import ApiError
from forge.models import Base
from forge.models_api import router


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    app = FastAPI()

    @app.exception_handler(ApiError)
    async def handle_api_error(request, exc):
        return JSONResponse(
            status_code=getattr(exc, "status_code", getattr(exc, "status", 500)),
            content={"error": str(exc)},
        )

    def session_override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    app.include_router(router)
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


VALID_MODEL = {
    "name": "Imported Test Model",
    "description": "Imported from JSON",
    "n_layer": 2,
    "d_model": 128,
    "n_head": 4,
    "ctx_len": 512,
    "vocab_size": 16384,
    "dropout": 0.0,
}


def test_import_and_retrieve_model(client):
    response = client.post("/models/import", json={"model": VALID_MODEL})
    assert response.status_code == 201, response.text
    model = response.json()
    assert model["name"] == VALID_MODEL["name"]
    assert model["param_count"] > 0
    fetched = client.get(f"/models/{model['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == VALID_MODEL["name"]


def test_import_without_wrapper(client):
    response = client.post("/models/import", json=VALID_MODEL)
    assert response.status_code == 201


def test_import_rejects_duplicate_name(client):
    assert client.post("/models/import", json=VALID_MODEL).status_code == 201
    assert client.post("/models/import", json=VALID_MODEL).status_code == 409


def test_import_rejects_bad_json(client):
    response = client.post(
        "/models/import", content=b"{invalid", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400


def test_import_rejects_missing_fields(client):
    response = client.post("/models/import", json={"model": {"name": "Incomplete"}})
    assert response.status_code == 422


def test_import_rejects_non_object(client):
    response = client.post("/models/import", content=json.dumps([1, 2]))
    assert response.status_code == 422


def test_import_rejects_oversized_body(client):
    response = client.post("/models/import", content=b" " * 1_048_577)
    assert response.status_code == 413
