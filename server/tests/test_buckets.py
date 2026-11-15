import json
import uuid
from pathlib import Path

from sqlalchemy import func, select

from forge.models import Dataset, Document, Job


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


def test_delete_empty_bucket_and_repeated_request(client):
    bucket = create(client, "temporary")
    summary = client.get(f"/api/buckets/{bucket['id']}/deletion-summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["bucket_name"] == "temporary"
    assert payload["document_count"] == 0
    assert payload["physical_file_count"] == 0
    assert payload["total_bytes"] == 0

    deleted = client.delete(
        f"/api/buckets/{bucket['id']}?confirm=true&scope_token={payload['scope_token']}"
    )
    assert deleted.status_code == 204
    assert client.get("/api/buckets").json()["total"] == 0

    repeated = client.delete(f"/api/buckets/{bucket['id']}?confirm=true")
    assert repeated.status_code == 404
    assert repeated.json()["error"]["code"] == "not_found"


def test_delete_requires_explicit_confirmation(client, make_document, data_dir):
    bucket = create(client, "busy")
    doc = make_document(uuid.UUID(bucket["id"]))
    raw_file = data_dir / "files" / doc.raw_uri
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_bytes(b"hello")

    response = client.delete(f"/api/buckets/{bucket['id']}")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "confirmation_required"
    assert client.get("/api/buckets").json()["total"] == 1
    assert raw_file.exists()


def test_delete_unknown_id_is_404(client):
    assert client.delete(f"/api/buckets/{uuid.uuid4()}").status_code == 404


def test_bad_uuid_in_path_is_422(client):
    assert client.delete("/api/buckets/not-a-uuid").status_code == 422


def test_deletion_summary_deduplicates_jsonl_raw_and_deletes_nonempty_bucket(client, db, data_dir):
    bucket = create(client, "nonempty-jsonl")
    sibling = create(client, "sibling")

    jsonl_payload = "\n".join(
        json.dumps({"text": f"Record body number {i} with enough clean text for storage."})
        for i in range(3)
    ).encode("utf-8")
    plain_payload = b"Plain text document stored alongside the JSONL batch."

    up = client.post(
        f"/api/buckets/{bucket['id']}/documents",
        files=[
            ("files", ("batch.jsonl", jsonl_payload, "application/x-ndjson")),
            ("files", ("plain.txt", plain_payload, "text/plain")),
        ],
        data={"source_note": "batch upload"},
    )
    assert up.status_code == 201, up.text
    sibling_up = client.post(
        f"/api/buckets/{sibling['id']}/documents",
        files=[("files", ("keep.txt", b"Sibling collection file must stay.", "text/plain"))],
    )
    assert sibling_up.status_code == 201

    summary_res = client.get(f"/api/buckets/{bucket['id']}/deletion-summary")
    assert summary_res.status_code == 200
    summary = summary_res.json()

    docs = db.scalars(select(Document).where(Document.bucket_id == uuid.UUID(bucket["id"]))).all()
    unique_raws = {doc.raw_uri for doc in docs if doc.raw_uri}
    unique_texts = {doc.text_uri for doc in docs if doc.text_uri}
    expected_raw_bytes = sum((data_dir / "files" / uri).stat().st_size for uri in unique_raws)
    expected_text_bytes = sum((data_dir / "files" / uri).stat().st_size for uri in unique_texts)

    assert summary["document_count"] == 4
    assert summary["raw_file_count"] == 2
    assert summary["text_file_count"] == 4
    assert summary["physical_file_count"] == 6
    assert summary["raw_bytes"] == expected_raw_bytes
    assert summary["text_bytes"] == expected_text_bytes
    assert summary["total_bytes"] == expected_raw_bytes + expected_text_bytes
    assert summary["shared_file_count"] == 0
    assert summary["missing_file_count"] == 0

    delete_res = client.delete(
        f"/api/buckets/{bucket['id']}?confirm=true&scope_token={summary['scope_token']}"
    )
    assert delete_res.status_code == 204
    assert client.get(f"/api/buckets/{bucket['id']}").status_code == 404
    assert (
        db.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.bucket_id == uuid.UUID(bucket["id"]))
        )
        == 0
    )

    for uri in unique_raws | unique_texts:
        assert not (data_dir / "files" / uri).exists()

    # Sibling collection and its files stay completely intact.
    sibling_docs = db.scalars(
        select(Document).where(Document.bucket_id == uuid.UUID(sibling["id"]))
    ).all()
    assert len(sibling_docs) == 1
    assert (data_dir / "files" / sibling_docs[0].raw_uri).exists()
    assert sibling_docs[0].text_uri and (data_dir / "files" / sibling_docs[0].text_uri).exists()


def test_delete_preserves_shared_raw_file_across_collections(client, db, data_dir):
    source = create(client, "source-shared")
    dest = create(client, "dest-shared")

    jsonl_payload = (
        json.dumps({"text": "Shared line one staying in source collection until deleted."})
        + "\n"
        + json.dumps({"text": "Shared line two moved into destination collection."})
    ).encode("utf-8")
    up = client.post(
        f"/api/buckets/{source['id']}/documents",
        files=[("files", ("shared.jsonl", jsonl_payload, "application/x-ndjson"))],
    )
    assert up.status_code == 201
    moved_id = up.json()["outcomes"][1]["document_id"]
    move_res = client.post(
        f"/api/documents/{moved_id}/move",
        json={"bucket_id": dest["id"]},
    )
    assert move_res.status_code == 200

    source_doc = db.scalar(select(Document).where(Document.bucket_id == uuid.UUID(source["id"])))
    dest_doc = db.scalar(select(Document).where(Document.bucket_id == uuid.UUID(dest["id"])))
    assert source_doc is not None and dest_doc is not None
    assert source_doc.raw_uri == dest_doc.raw_uri

    shared_raw_path = data_dir / "files" / dest_doc.raw_uri
    dest_text_path = data_dir / "files" / (dest_doc.text_uri or "")
    source_text_path = data_dir / "files" / (source_doc.text_uri or "")

    summary = client.get(f"/api/buckets/{source['id']}/deletion-summary").json()
    assert summary["document_count"] == 1
    assert summary["shared_file_count"] == 1
    assert summary["raw_file_count"] == 0
    assert summary["raw_bytes"] == 0
    assert summary["text_file_count"] == 1
    assert summary["total_bytes"] == source_text_path.stat().st_size

    res = client.delete(
        f"/api/buckets/{source['id']}?confirm=true&scope_token={summary['scope_token']}"
    )
    assert res.status_code == 204
    assert not source_text_path.exists()
    assert shared_raw_path.exists()
    assert dest_text_path.exists()
    assert client.get(f"/api/documents/{moved_id}/text").status_code == 200


def test_delete_bulk_collection_with_thousands_of_files_and_preserve_snapshot(
    client, db, data_dir, make_dataset, make_model, make_run, make_checkpoint
):
    bucket = create(client, "massive-corpus")
    bucket_uuid = uuid.UUID(bucket["id"])
    files_dir = data_dir / "files"
    (files_dir / "raw").mkdir(parents=True, exist_ok=True)
    (files_dir / "text").mkdir(parents=True, exist_ok=True)

    raw_uris = []
    for idx in range(50):
        rel = f"raw/shared-batch-{idx}"
        (files_dir / rel).write_bytes(f"raw-batch-{idx}".encode())
        raw_uris.append(rel)

    docs = []
    for idx in range(2100):
        doc_id = uuid.uuid4()
        text_rel = f"text/{doc_id}.txt"
        (files_dir / text_rel).write_bytes(b"clean text payload\n")
        docs.append(
            Document(
                id=doc_id,
                bucket_id=bucket_uuid,
                filename=f"doc-{idx}.jsonl",
                source_format="jsonl",
                source_note="",
                raw_uri=raw_uris[idx % len(raw_uris)],
                text_uri=text_rel,
                status="parsed",
                content_hash=f"hash-{idx}",
                char_count=19,
                word_count=3,
                quality={"flags": []},
            )
        )
    db.add_all(docs)
    db.commit()

    # Create independent snapshot + downstream training artifacts to confirm preservation.
    snapshot = make_dataset(name="frozen-snap", bucket_id=bucket_uuid, status="ready")
    snap_file = data_dir / "datasets" / str(snapshot.id) / "train.jsonl"
    snap_file.parent.mkdir(parents=True, exist_ok=True)
    snap_file.write_text('{"text": "frozen"}\n', encoding="utf-8")

    model = make_model(name="kept-model")
    run = make_run(model_id=model.id)
    ckpt = make_checkpoint(run_id=run.id)
    ckpt_file = data_dir / ckpt.uri
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    ckpt_file.write_bytes(b"checkpoint-weights")

    summary = client.get(f"/api/buckets/{bucket['id']}/deletion-summary").json()
    assert summary["document_count"] == 2100
    assert summary["raw_file_count"] == 50
    assert summary["text_file_count"] == 2100
    assert summary["physical_file_count"] == 2150

    deleted = client.delete(
        f"/api/buckets/{bucket['id']}?confirm=true&scope_token={summary['scope_token']}"
    )
    assert deleted.status_code == 204
    assert client.get(f"/api/buckets/{bucket['id']}").status_code == 404
    assert (
        db.scalar(
            select(func.count()).select_from(Document).where(Document.bucket_id == bucket_uuid)
        )
        == 0
    )
    for rel in raw_uris:
        assert not (files_dir / rel).exists()
    for doc in docs:
        assert doc.text_uri and not (files_dir / doc.text_uri).exists()

    db.expire_all()
    kept_snapshot = db.get(Dataset, snapshot.id)
    assert kept_snapshot is not None
    assert kept_snapshot.bucket_id is None
    assert snap_file.exists()
    assert ckpt_file.exists()


def test_delete_missing_files_and_filesystem_failure_allows_retry(
    client, db, data_dir, monkeypatch
):
    bucket = create(client, "retry-cleanup")
    up = client.post(
        f"/api/buckets/{bucket['id']}/documents",
        files=[
            ("files", ("first.txt", b"First file bytes for retry check.", "text/plain")),
            ("files", ("second.txt", b"Second file bytes for retry check.", "text/plain")),
        ],
    )
    assert up.status_code == 201

    docs = db.scalars(select(Document).where(Document.bucket_id == uuid.UUID(bucket["id"]))).all()
    missing_raw = data_dir / "files" / docs[0].raw_uri
    missing_raw.unlink()
    locked_text = (data_dir / "files" / (docs[1].text_uri or "")).resolve()

    summary1 = client.get(f"/api/buckets/{bucket['id']}/deletion-summary").json()
    assert summary1["missing_file_count"] == 1
    assert summary1["physical_file_count"] == 3

    real_unlink = Path.unlink

    def failing_unlink(self: Path, *args, **kwargs):
        if self.resolve() == locked_text:
            raise PermissionError("locked by another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    failed = client.delete(
        f"/api/buckets/{bucket['id']}?confirm=true&scope_token={summary1['scope_token']}"
    )
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "cleanup_failed"
    assert "locked by another process" in failed.json()["error"]["message"]
    assert client.get(f"/api/buckets/{bucket['id']}").status_code == 200
    assert locked_text.exists()

    # Release simulated file lock and retry deletion using refreshed summary.
    monkeypatch.undo()
    summary2 = client.get(f"/api/buckets/{bucket['id']}/deletion-summary").json()
    assert summary2["physical_file_count"] == 1
    assert summary2["missing_file_count"] == 3

    retried = client.delete(
        f"/api/buckets/{bucket['id']}?confirm=true&scope_token={summary2['scope_token']}"
    )
    assert retried.status_code == 204
    assert not locked_text.exists()
    assert client.get(f"/api/buckets/{bucket['id']}").status_code == 404


def test_delete_blocks_active_jobs_before_destructive_work(client, db, data_dir):
    bucket = create(client, "active-job")
    up = client.post(
        f"/api/buckets/{bucket['id']}/documents",
        files=[("files", ("doc.txt", b"Document protected while jobs run.", "text/plain"))],
    )
    assert up.status_code == 201
    doc = db.scalar(select(Document).where(Document.bucket_id == uuid.UUID(bucket["id"])))
    assert doc is not None
    raw_path = data_dir / "files" / doc.raw_uri

    import_job = Job(
        job_type="dataset_import",
        status="running",
        payload={"target_id": bucket["id"], "target_name": bucket["name"]},
        progress={},
    )
    db.add(import_job)
    db.commit()

    summary = client.get(f"/api/buckets/{bucket['id']}/deletion-summary").json()
    assert len(summary["active_jobs"]) == 1
    assert summary["active_jobs"][0]["job_type"] == "dataset_import"

    blocked_import = client.delete(f"/api/buckets/{bucket['id']}?confirm=true")
    assert blocked_import.status_code == 409
    assert blocked_import.json()["error"]["code"] == "bucket_busy"
    assert raw_path.exists()

    import_job.status = "completed"
    snapshot_job = Job(
        job_type="snapshot_create",
        status="pending",
        payload={"target_id": str(uuid.uuid4()), "bucket_id": bucket["id"]},
        progress={},
    )
    db.add(snapshot_job)
    db.commit()

    blocked_snapshot = client.delete(f"/api/buckets/{bucket['id']}?confirm=true")
    assert blocked_snapshot.status_code == 409
    assert blocked_snapshot.json()["error"]["code"] == "bucket_busy"
    assert raw_path.exists()

    snapshot_job.status = "cancelled"
    db.commit()
    assert client.delete(f"/api/buckets/{bucket['id']}?confirm=true").status_code == 204


def test_delete_detects_collection_change_after_summary(client, db, data_dir):
    bucket = create(client, "changing-collection")
    first_up = client.post(
        f"/api/buckets/{bucket['id']}/documents",
        files=[("files", ("one.txt", b"Initial file content here.", "text/plain"))],
    )
    assert first_up.status_code == 201
    first_summary = client.get(f"/api/buckets/{bucket['id']}/deletion-summary").json()
    assert first_summary["document_count"] == 1

    second_up = client.post(
        f"/api/buckets/{bucket['id']}/documents",
        files=[("files", ("two.txt", b"Added after modal opened.", "text/plain"))],
    )
    assert second_up.status_code == 201
    docs_before = db.scalars(
        select(Document).where(Document.bucket_id == uuid.UUID(bucket["id"]))
    ).all()
    assert len(docs_before) == 2

    stale = client.delete(
        f"/api/buckets/{bucket['id']}?confirm=true&scope_token={first_summary['scope_token']}"
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "scope_changed"
    assert client.get(f"/api/buckets/{bucket['id']}").status_code == 200
    for doc in docs_before:
        assert (data_dir / "files" / doc.raw_uri).exists()
        assert doc.text_uri and (data_dir / "files" / doc.text_uri).exists()

    second_summary = client.get(f"/api/buckets/{bucket['id']}/deletion-summary").json()
    assert second_summary["document_count"] == 2
    assert second_summary["scope_token"] != first_summary["scope_token"]

    renamed = client.patch(f"/api/buckets/{bucket['id']}", json={"name": "renamed-collection"})
    assert renamed.status_code == 200
    stale_after_rename = client.delete(
        f"/api/buckets/{bucket['id']}?confirm=true&scope_token={second_summary['scope_token']}"
    )
    assert stale_after_rename.status_code == 409
    assert stale_after_rename.json()["error"]["code"] == "scope_changed"
    for doc in docs_before:
        assert (data_dir / "files" / doc.raw_uri).exists()
        assert doc.text_uri and (data_dir / "files" / doc.text_uri).exists()

    third_summary = client.get(f"/api/buckets/{bucket['id']}/deletion-summary").json()
    assert third_summary["bucket_name"] == "renamed-collection"
    assert third_summary["scope_token"] != second_summary["scope_token"]

    confirmed = client.delete(
        f"/api/buckets/{bucket['id']}?confirm=true&scope_token={third_summary['scope_token']}"
    )
    assert confirmed.status_code == 204


def test_delete_blocks_new_conflicting_work_during_deletion(client, db, data_dir, monkeypatch):
    from forge import buckets as buckets_module
    from forge.storage import acquire_bucket_lock

    bucket = create(client, "locked-during-delete")
    up = client.post(
        f"/api/buckets/{bucket['id']}/documents",
        files=[
            ("files", ("a.txt", b"First document in locked deletion test.", "text/plain")),
            ("files", ("b.txt", b"Second document in locked deletion test.", "text/plain")),
        ],
    )
    assert up.status_code == 201
    summary = client.get(f"/api/buckets/{bucket['id']}/deletion-summary").json()

    monkeypatch.setattr(buckets_module, "validate_import_params", lambda *args, **kwargs: None)
    real_delete_storage_files = buckets_module.delete_storage_files
    mid_delete_responses: list[tuple[str, int, str | None]] = []

    def intercepted_delete(uris):
        import_res = client.post(
            f"/api/buckets/{bucket['id']}/import",
            json={
                "dataset": "org/ds",
                "config": "default",
                "split": "train",
                "text_column": "text",
                "limit": 10,
            },
        )
        mid_delete_responses.append(
            ("import", import_res.status_code, import_res.json().get("error", {}).get("code"))
        )
        snap_res = client.post(
            "/api/datasets",
            json={"name": "concurrent-snap", "bucket_id": bucket["id"]},
        )
        mid_delete_responses.append(
            ("snapshot", snap_res.status_code, snap_res.json().get("error", {}).get("code"))
        )
        upload_res = client.post(
            f"/api/buckets/{bucket['id']}/documents",
            files=[("files", ("c.txt", b"Should be blocked mid-delete.", "text/plain"))],
        )
        mid_delete_responses.append(
            ("upload", upload_res.status_code, upload_res.json().get("error", {}).get("code"))
        )
        return real_delete_storage_files(uris)

    # Holding the collection lock externally also blocks delete_bucket before destructive work.
    with acquire_bucket_lock(uuid.UUID(bucket["id"]), timeout=0):
        busy_del = client.delete(
            f"/api/buckets/{bucket['id']}?confirm=true&scope_token={summary['scope_token']}"
        )
        assert busy_del.status_code == 409
        assert busy_del.json()["error"]["code"] == "bucket_busy"

    monkeypatch.setattr(buckets_module, "delete_storage_files", intercepted_delete)
    res = client.delete(
        f"/api/buckets/{bucket['id']}?confirm=true&scope_token={summary['scope_token']}"
    )
    assert res.status_code == 204
    assert mid_delete_responses == [
        ("import", 409, "bucket_busy"),
        ("snapshot", 409, "bucket_busy"),
        ("upload", 409, "bucket_busy"),
    ]
    assert db.scalar(select(func.count()).select_from(Job)) == 0
    assert db.scalar(select(func.count()).select_from(Dataset)) == 0


def test_delete_rejects_path_escape_and_preserves_database(client, db, data_dir):
    bucket = create(client, "unsafe-path")
    db_file = data_dir / "test.db"
    db_before = db_file.exists()
    assert db_before

    evil_doc = Document(
        bucket_id=uuid.UUID(bucket["id"]),
        filename="escape.txt",
        source_format="txt",
        source_note="",
        raw_uri="../test.db",
        text_uri=None,
        status="parsed",
        content_hash="escape-hash",
        char_count=5,
        word_count=1,
        quality={},
    )
    db.add(evil_doc)
    db.commit()

    summary_res = client.get(f"/api/buckets/{bucket['id']}/deletion-summary")
    assert summary_res.status_code == 409
    assert summary_res.json()["error"]["code"] == "invalid_storage_path"

    delete_res = client.delete(f"/api/buckets/{bucket['id']}?confirm=true")
    assert delete_res.status_code == 409
    assert delete_res.json()["error"]["code"] == "invalid_storage_path"
    assert db_file.exists() == db_before
    assert data_dir.exists()
