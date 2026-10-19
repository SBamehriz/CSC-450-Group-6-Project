
import uuid

from forge.models import Job


def test_create_training_job(client, db, make_dataset):
    dataset = make_dataset()

    response = client.post(
        "/api/jobs",
        json={
            "dataset_id": str(dataset.id),
            "preset": "nano",
            "learning_rate": 0.001,
            "epochs": 1,
        },
    )

    assert response.status_code == 201, response.text

    result = response.json()
    assert "id" in result
    assert result["status"] == "pending"

    job = db.get(Job, uuid.UUID(result["id"]))
    assert job is not None
    assert job.job_type == "training"
    assert job.status == "pending"

def test_training_job_invalid_dataset(client):
    response = client.post(
        "/api/jobs",
        json={
            "dataset_id": str(uuid.uuid4()),
            "preset": "nano",
            "learning_rate": 0.001,
            "epochs": 1,
        },
    )

    assert response.status_code == 404
    print("status code:", response.status_code)


def test_training_job_invalid_preset(client, make_dataset):
    dataset = make_dataset()

    response = client.post(
        "/api/jobs",
        json={
            "dataset_id": str(dataset.id),
            "preset": "invalid_preset",
            "learning_rate": 0.001,
            "epochs": 1,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["message"] == "Invalid training preset"
