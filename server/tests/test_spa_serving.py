import pytest
from fastapi.testclient import TestClient

from forge import main


@pytest.fixture()
def dist_dir(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Forge</title>")
    (dist / "assets" / "app.js").write_text("console.log('hi')")
    (dist / "favicon.svg").write_text("<svg/>")
    return dist


@pytest.fixture()
def spa_client(engine, db, dist_dir, monkeypatch) -> TestClient:
    from forge import db as db_module

    db_module.reset_engine()
    monkeypatch.setattr(main, "WEB_DIST", dist_dir)
    return TestClient(main.create_app())


def test_unknown_api_route_is_json_not_html(spa_client):
    response = spa_client.get("/api/nope")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "not_found"


def test_real_api_routes_still_work(spa_client):
    assert spa_client.get("/api/buckets").status_code == 200
    assert spa_client.get("/api/health").status_code == 200


def test_client_side_routes_get_index_html(spa_client):
    for path in ("/", "/data", "/models/some-id"):
        response = spa_client.get(path)
        assert response.status_code == 200
        assert "<title>Forge</title>" in response.text


def test_real_files_are_served(spa_client):
    assert spa_client.get("/favicon.svg").text == "<svg/>"
    assert spa_client.get("/assets/app.js").status_code == 200


def test_without_a_build_there_is_no_catch_all(engine, db, tmp_path, monkeypatch):
    from forge import db as db_module

    db_module.reset_engine()
    monkeypatch.setattr(main, "WEB_DIST", tmp_path / "does-not-exist")
    client = TestClient(main.create_app())

    assert client.get("/data").status_code == 404
    assert client.get("/api/nope").json()["error"]["code"] == "not_found"
