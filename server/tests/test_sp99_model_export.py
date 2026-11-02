import uuid


def test_export_model_success(client, make_model):
    """A saved model can be exported as JSON."""
    model = make_model(name="Export Test Model")

    response = client.get(f"/api/models/{model.id}/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    data = response.json()
    assert data["name"] == "Export Test Model"
    assert data["description"] == model.description
    assert data["preset"] == model.preset


def test_export_model_configuration(client, make_model):
    """The exported JSON contains the model's architecture."""
    model = make_model()

    response = client.get(f"/api/models/{model.id}/export")
    data = response.json()

    assert data["n_layer"] == model.n_layer
    assert data["d_model"] == model.d_model
    assert data["n_head"] == model.n_head
    assert data["ctx_len"] == model.ctx_len
    assert data["vocab_size"] == model.vocab_size
    assert data["dropout"] == model.dropout
    assert data["config"] == model.config


def test_export_model_download_filename(client, make_model):
    """The response provides a downloadable JSON filename."""
    model = make_model()

    response = client.get(f"/api/models/{model.id}/export")

    assert response.status_code == 200
    assert response.headers["content-disposition"] == (
        f'attachment; filename="model-{model.id}.json"'
    )


def test_export_model_not_found(client):
    """Exporting a nonexistent model returns HTTP 404."""
    missing_id = uuid.uuid4()

    response = client.get(f"/api/api/models/{missing_id}/export")

    assert response.status_code == 404


def test_export_model_invalid_id(client):
    """An invalid model ID is rejected."""
    response = client.get("/api/models/not-a-valid-uuid/export")

    assert response.status_code == 422
