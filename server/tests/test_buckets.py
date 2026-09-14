import uuid


def create(client, name: str, description: str = "") -> dict:
    response = client.post("/api/buckets", json={"name": name, "description": description})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_returns_the_bucket_with_zero_stats(client):
    body = create(client, "gutenberg-fiction", "public domain books")
    assert body["name"] == "gutenberg-fiction"
    assert body["description"] == "public domain books"
    assert body["stats"] == {
        "documents": 0,
        "parsed_documents": 0,
        "chars": 0,
        "est_tokens": 0,
    }
    uuid.UUID(body["id"])


def test_timestamps_come_back_as_utc(client):
    created = create(client, "timezones")["created_at"]
    assert created.endswith("Z") or "+00:00" in created, created

    listed = client.get("/api/buckets").json()["items"][0]["created_at"]
    assert listed.endswith("Z") or "+00:00" in listed, listed
    assert listed == created


def test_names_are_unique(client):
    create(client, "caselaw")
    response = client.post("/api/buckets", json={"name": "caselaw"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "name_taken"


def test_names_are_trimmed(client):
    assert create(client, "  simple-wikipedia  ")["name"] == "simple-wikipedia"


def test_list_is_sorted_by_name_and_counts_total(client):
    for name in ["zeta", "alpha", "mid"]:
        create(client, name)
    body = client.get("/api/buckets").json()
    assert body["total"] == 3
    assert [b["name"] for b in body["items"]] == ["alpha", "mid", "zeta"]


def test_pagination(client):
    for i in range(5):
        create(client, f"bucket-{i}")
    body = client.get("/api/buckets?limit=2&offset=2").json()
    assert body["total"] == 5
    assert [b["name"] for b in body["items"]] == ["bucket-2", "bucket-3"]


def test_limit_is_capped(client):
    assert client.get("/api/buckets?limit=500").status_code == 422


def test_stats_count_documents(client, make_document):
    bucket = create(client, "with-docs")
    bucket_id = uuid.UUID(bucket["id"])
    make_document(bucket_id, chars=1000)
    make_document(bucket_id, chars=2000)
    make_document(bucket_id, chars=500, status="failed")

    body = client.get("/api/buckets").json()
    stats = body["items"][0]["stats"]
    assert stats["documents"] == 3
    assert stats["parsed_documents"] == 2
    assert stats["chars"] == 3500
    assert stats["est_tokens"] == 875


def test_empty_bucket_has_zero_stats_not_nulls(client, make_document):
    full = create(client, "aaa-full")
    create(client, "bbb-empty")
    make_document(uuid.UUID(full["id"]), chars=40)

    items = client.get("/api/buckets").json()["items"]
    assert items[0]["stats"]["chars"] == 40
    assert items[1]["stats"] == {
        "documents": 0,
        "parsed_documents": 0,
        "chars": 0,
        "est_tokens": 0,
    }


def test_patch_updates_fields(client):
    bucket = create(client, "old-name", "old note")
    response = client.patch(f"/api/buckets/{bucket['id']}", json={"description": "new note"})
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "old-name"
    assert body["description"] == "new note"


def test_patch_rejects_a_taken_name(client):
    create(client, "taken")
    mine = create(client, "mine")
    response = client.patch(f"/api/buckets/{mine['id']}", json={"name": "taken"})
    assert response.status_code == 409


def test_patch_to_its_own_name_is_fine(client):
    bucket = create(client, "same")
    response = client.patch(f"/api/buckets/{bucket['id']}", json={"name": "same"})
    assert response.status_code == 200


def test_patch_unknown_id_is_404(client):
    response = client.patch(f"/api/buckets/{uuid.uuid4()}", json={"name": "x"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_delete_empty_bucket(client):
    bucket = create(client, "temporary")
    assert client.delete(f"/api/buckets/{bucket['id']}").status_code == 204
    assert client.get("/api/buckets").json()["total"] == 0


def test_delete_refuses_a_bucket_with_documents(client, make_document):
    bucket = create(client, "busy")
    make_document(uuid.UUID(bucket["id"]))
    response = client.delete(f"/api/buckets/{bucket['id']}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "bucket_not_empty"
    assert client.get("/api/buckets").json()["total"] == 1


def test_delete_unknown_id_is_404(client):
    assert client.delete(f"/api/buckets/{uuid.uuid4()}").status_code == 404


def test_bad_uuid_in_path_is_422(client):
    assert client.delete("/api/buckets/not-a-uuid").status_code == 422
