
from xml.parsers.expat import model

from fastapi.testclient import TestClient
from sqlalchemy import select
from forge.models import Bucket

def test_delete_all_models_requires_confirmation(client: TestClient):
    """Bulk deletion must reject requests without confirmation."""
    response = client.delete("/api/models/all")

    assert response.status_code == 400
    assert response.json()["detail"] == "Confirmation required"


def test_delete_all_models_empty_database(client: TestClient):
    response = client.delete("/api/models/all?confirm=true")
    assert response.status_code == 204


def test_delete_all_models_does_not_delete_unrelated_data(client: TestClient, db):
    from forge.models import Bucket, Model
    from sqlalchemy import select
    import uuid

    bucket = Bucket(name=f"keep-{uuid.uuid4()}")
    model = Model(
        name=f"remove-{uuid.uuid4()}", n_layer=2, d_model=128,
        n_head=4, ctx_len=512, vocab_size=4096,
    )
    db.add_all([bucket, model])
    db.commit()
    model_id = model.id
    bucket_id = bucket.id
    response = client.delete("/api/models/all?confirm=true")
    assert response.status_code == 204
    db.expire_all()
    assert db.scalar(select(Model).where(Model.id == model_id)) is None
    assert db.scalar(select(Bucket).where(Bucket.id == bucket_id)) is not None


def test_delete_all_models_rejects_referenced_models(client: TestClient, db):
    from forge.models import Model, Run
    from sqlalchemy import select
    import uuid

    model = Model(
        name=f"protected-{uuid.uuid4()}", n_layer=2, d_model=128,
        n_head=4, ctx_len=512, vocab_size=4096,
    )
    db.add(model)
    db.flush()
    run = Run(name="keep-run", model_id=model.id)
    db.add(run)


    db.commit()
    model_id = model.id
    run_id = run.id

    response = client.delete("/api/models/all?confirm=true")
    assert response.status_code == 409
    db.expire_all()
    assert db.scalar(select(Model).where(Model.id == model_id)) is not None
    assert db.scalar(select(Run).where(Run.id == run_id)) is not None

    assert db.scalar(select(Run).where(Run.id == run.id)) is not None
