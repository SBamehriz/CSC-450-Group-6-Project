import gzip
import io
import uuid
from pathlib import Path

import pytest

from forge import documents as documents_router
from tests.conftest import ARTICLE_PDF, NOTES_MD, SCANNED_PDF

FIXTURES = Path(__file__).parent / "fixtures"

TEXT = b"A perfectly ordinary paragraph of English prose, long enough to clear the flags. " * 12


def upload(client, bucket_id, files, source_note=""):
    payload = [
        ("files", (name, io.BytesIO(data), "application/octet-stream")) for name, data in files
    ]
    return client.post(
        f"/api/buckets/{bucket_id}/documents",
        files=payload,
        data={"source_note": source_note},
    )


@pytest.fixture()
def bucket(client):
    response = client.post("/api/buckets", json={"name": "corpus"})
    return response.json()["id"]


@pytest.fixture()
def other_bucket(client):
    response = client.post("/api/buckets", json={"name": "elsewhere"})
    return response.json()["id"]


def test_upload_one_text_file(client, bucket):
    response = upload(client, bucket, [("a.txt", TEXT)], source_note="scraped Aug 2026")
    assert response.status_code == 201
    body = response.json()
    assert body == {
        "parsed": 1,
        "failed": 0,
        "outcomes": [
            {
                "filename": "a.txt",
                "status": "parsed",
                "document_id": body["outcomes"][0]["document_id"],
                "error": None,
            }
        ],
    }

    document = client.get(f"/api/documents/{body['outcomes'][0]['document_id']}").json()
    assert document["status"] == "parsed"
    assert document["source_format"] == "txt"
    assert document["source_note"] == "scraped Aug 2026"
    assert document["char_count"] > 0
    assert document["content_hash"]
    assert document["quality"]["flags"] == []


def test_upload_every_supported_format_at_once(client, bucket):
    response = upload(
        client,
        bucket,
        [
            ("plain.txt", (FIXTURES / "plain.txt").read_bytes()),
            ("notes.md", NOTES_MD),
            ("article.html", (FIXTURES / "article.html").read_bytes()),
            ("article.pdf", ARTICLE_PDF),
            ("records.jsonl", (FIXTURES / "records.jsonl").read_bytes()),
            ("compressed.txt.gz", (FIXTURES / "compressed.txt.gz").read_bytes()),
        ],
    )
    body = response.json()
    # jsonl gives 3, the rest 1 each
    assert body["parsed"] == 8, body
    assert body["failed"] == 0

    formats = {d["source_format"] for d in client.get("/api/documents").json()["items"]}
    assert formats == {"txt", "md", "html", "pdf", "jsonl"}


def test_one_bad_file_does_not_spoil_the_batch(client, bucket):
    body = upload(
        client,
        bucket,
        [
            ("good.txt", TEXT),
            ("scanned.pdf", SCANNED_PDF),
            ("thesis.docx", b"not a document we read"),
            ("also-good.txt", TEXT.replace(b"ordinary", b"different")),
        ],
    ).json()

    assert body["parsed"] == 2
    assert body["failed"] == 2
    by_name = {o["filename"]: o for o in body["outcomes"]}
    assert "scanned" in by_name["scanned.pdf"]["error"].lower()
    assert "OCR" in by_name["scanned.pdf"]["error"]
    assert ".docx" in by_name["thesis.docx"]["error"]


def test_an_unreadable_file_records_what_it_actually_was(client, bucket):
    upload(client, bucket, [("thesis.docx", b"PK not a document we read")])
    document = client.get("/api/documents").json()["items"][0]
    assert document["source_format"] == "docx"
    assert document["status"] == "failed"


def test_an_unsupported_file_records_as_other(client, bucket):
    upload(client, bucket, [("data.csv", b"col1,col2\n1,2")])
    document = client.get("/api/documents").json()["items"][0]
    assert document["source_format"] == "other"
    assert document["status"] == "failed"


def test_upload_valid_docx_and_image(client, bucket):
    import io
    import docx
    from PIL import Image, ImageDraw

    doc = docx.Document()
    doc.add_paragraph("A perfectly valid docx paragraph long enough to pass cleaning standards. " * 5)
    buf_docx = io.BytesIO()
    doc.save(buf_docx)

    img = Image.new("RGB", (400, 120), color="white")
    d = ImageDraw.Draw(img)
    d.text((10, 40), "Uploaded Invoice Scan 2026", fill="black")
    buf_img = io.BytesIO()
    img.save(buf_img, format="PNG")

    res = upload(
        client,
        bucket,
        [
            ("paper.docx", buf_docx.getvalue()),
            ("receipt.png", buf_img.getvalue()),
        ],
    )
    assert res.status_code == 201
    body = res.json()
    assert body["parsed"] == 2
    assert body["failed"] == 0

    docs = client.get("/api/documents").json()["items"]
    formats = {d["source_format"] for d in docs}
    assert "docx" in formats
    assert "image" in formats


def test_failed_files_are_kept_as_rows_so_you_can_see_them(client, bucket):
    upload(client, bucket, [("scanned.pdf", SCANNED_PDF)])
    failed = client.get("/api/documents?status=failed").json()
    assert failed["total"] == 1
    assert "scanned" in failed["items"][0]["error"].lower()


def test_jsonl_line_numbers_are_kept_in_the_filename(client, bucket):
    upload(client, bucket, [("records.jsonl", (FIXTURES / "records.jsonl").read_bytes())])
    names = [d["filename"] for d in client.get("/api/documents").json()["items"]]
    assert "records.jsonl#L1" in names
    assert "records.jsonl#L6" in names


def test_quality_flags_reach_the_api(client, bucket):
    upload(client, bucket, [("tiny.txt", b"too short to be useful")])
    document = client.get("/api/documents").json()["items"][0]
    assert "too_short" in document["quality"]["flags"]
    assert document["quality"]["alpha_ratio"] > 0


def test_upload_to_a_missing_bucket_is_404(client):
    response = upload(client, uuid.uuid4(), [("a.txt", TEXT)])
    assert response.status_code == 404


def test_identical_text_is_refused_as_a_duplicate(client, bucket):
    upload(client, bucket, [("first.txt", TEXT)])
    body = upload(client, bucket, [("second.txt", TEXT)]).json()

    assert body["failed"] == 1
    assert "Duplicate of first.txt" in body["outcomes"][0]["error"]


def test_duplicate_detection_survives_whitespace_differences(client, bucket):
    upload(client, bucket, [("first.txt", b"Hello world.\n\n\n\nSecond line.")])
    body = upload(client, bucket, [("second.txt", b"Hello world.\n\nSecond line.   ")]).json()
    # cleaning makes both hash the same
    assert body["failed"] == 1


def test_the_same_text_in_a_different_bucket_is_fine(client, bucket, other_bucket):
    assert upload(client, bucket, [("a.txt", TEXT)]).json()["parsed"] == 1
    assert upload(client, other_bucket, [("a.txt", TEXT)]).json()["parsed"] == 1


def test_filter_by_bucket_status_and_name(client, bucket, other_bucket):
    upload(client, bucket, [("alpha.txt", TEXT)])
    upload(client, bucket, [("scanned.pdf", SCANNED_PDF)])
    upload(client, other_bucket, [("beta.txt", TEXT)])

    assert client.get(f"/api/documents?bucket_id={bucket}").json()["total"] == 2
    assert client.get("/api/documents?status=parsed").json()["total"] == 2
    assert client.get("/api/documents?status=failed").json()["total"] == 1
    assert client.get("/api/documents?q=alph").json()["total"] == 1


def test_filter_by_flagged(client, bucket):
    upload(client, bucket, [("clean.txt", TEXT), ("tiny.txt", b"short")])
    flagged = client.get("/api/documents?flagged=true").json()
    assert flagged["total"] == 1
    assert flagged["items"][0]["filename"] == "tiny.txt"


def test_flagged_shows_parsed_documents_only(client, bucket):
    upload(client, bucket, [("tiny.txt", b"short")])
    upload(client, bucket, [("tiny-again.txt", b"short")])
    flagged = client.get("/api/documents?flagged=true").json()
    assert flagged["total"] == 1
    assert flagged["items"][0]["status"] == "parsed"


def test_document_pagination(client, bucket):
    upload(client, bucket, [(f"f{i}.txt", TEXT + str(i).encode()) for i in range(5)])
    page = client.get("/api/documents?limit=2&offset=2").json()
    assert page["total"] == 5
    assert len(page["items"]) == 2


def test_document_detail_404(client):
    assert client.get(f"/api/documents/{uuid.uuid4()}").status_code == 404


def test_preview_window(client, bucket):
    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    document_id = body["outcomes"][0]["document_id"]

    whole = client.get(f"/api/documents/{document_id}/text").json()
    assert whole["offset"] == 0
    assert whole["total"] > 0
    assert whole["text"].startswith("A perfectly ordinary")

    window = client.get(f"/api/documents/{document_id}/text?offset=10&length=20").json()
    assert window["length"] == 20
    assert window["text"] == whole["text"][10:30]
    assert window["total"] == whole["total"]


def test_preview_past_the_end_is_empty_not_an_error(client, bucket):
    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    window = client.get(
        f"/api/documents/{body['outcomes'][0]['document_id']}/text?offset=999999",
    ).json()
    assert window["text"] == ""


def test_preview_window_is_capped(client, bucket):
    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    response = client.get(
        f"/api/documents/{body['outcomes'][0]['document_id']}/text?length=999999999",
    )
    assert response.status_code == 422


def test_a_failed_document_has_no_text_to_preview(client, bucket):
    upload(client, bucket, [("scanned.pdf", SCANNED_PDF)])
    document = client.get("/api/documents?status=failed").json()["items"][0]
    response = client.get(f"/api/documents/{document['id']}/text")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "no_text"


def test_move_between_buckets(client, bucket, other_bucket):
    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    document_id = body["outcomes"][0]["document_id"]

    moved = client.post(f"/api/documents/{document_id}/move", json={"bucket_id": other_bucket})
    assert moved.status_code == 200
    assert moved.json()["bucket_id"] == other_bucket
    assert client.get(f"/api/documents?bucket_id={bucket}").json()["total"] == 0


def test_move_into_a_bucket_that_already_has_that_text_is_refused(client, bucket, other_bucket):
    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    upload(client, other_bucket, [("copy.txt", TEXT)])

    response = client.post(
        f"/api/documents/{body['outcomes'][0]['document_id']}/move",
        json={"bucket_id": other_bucket},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "duplicate_in_target"


def test_move_to_a_missing_bucket_is_404(client, bucket):
    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    response = client.post(
        f"/api/documents/{body['outcomes'][0]['document_id']}/move",
        json={"bucket_id": str(uuid.uuid4())},
    )
    assert response.status_code == 404


def test_reject_keeps_the_row_and_the_reason(client, bucket):
    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    document_id = body["outcomes"][0]["document_id"]

    rejected = client.post(
        f"/api/documents/{document_id}/reject",
        json={"reason": "mostly a table of citations"},
    ).json()
    assert rejected["status"] == "rejected"
    assert rejected["error"] == "mostly a table of citations"
    assert client.get(f"/api/documents/{document_id}").status_code == 200


def test_rejecting_frees_the_text_for_a_better_copy(client, bucket):
    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    client.post(
        f"/api/documents/{body['outcomes'][0]['document_id']}/reject",
        json={"reason": "bad extraction"},
    )
    assert upload(client, bucket, [("better.txt", TEXT)]).json()["parsed"] == 1


def test_delete_removes_the_row(client, bucket):
    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    document_id = body["outcomes"][0]["document_id"]
    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert client.get(f"/api/documents/{document_id}").status_code == 404


def test_delete_unknown_id_is_404(client):
    assert client.delete(f"/api/documents/{uuid.uuid4()}").status_code == 404


def test_bucket_stats_follow_the_documents(client, bucket):
    def stats():
        buckets = client.get("/api/buckets").json()["items"]
        return next(b["stats"] for b in buckets if b["id"] == bucket)

    assert stats() == {"documents": 0, "parsed_documents": 0, "chars": 0, "est_tokens": 0}

    body = upload(client, bucket, [("a.txt", TEXT)]).json()
    after_upload = stats()
    assert after_upload["documents"] == 1
    assert after_upload["parsed_documents"] == 1
    assert after_upload["chars"] > 0
    assert after_upload["est_tokens"] == after_upload["chars"] // 4

    client.delete(f"/api/documents/{body['outcomes'][0]['document_id']}")
    assert stats()["documents"] == 0


def test_a_bucket_with_documents_still_refuses_deletion(client, bucket):
    upload(client, bucket, [("a.txt", TEXT)])
    response = client.delete(f"/api/buckets/{bucket}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "bucket_not_empty"


def test_the_documented_limits_are_what_the_code_uses():
    assert documents_router.MAX_FILE_BYTES == 50 * 1024 * 1024
    assert documents_router.MAX_FILES_PER_REQUEST == 200


def test_a_file_over_the_size_limit_is_refused(client, bucket, monkeypatch):
    # dont upload 50MB in ci
    monkeypatch.setattr(documents_router, "MAX_FILE_BYTES", 100)
    body = upload(client, bucket, [("big.txt", b"x" * 500)]).json()
    assert body["failed"] == 1
    assert "over the" in body["outcomes"][0]["error"]
    assert client.get("/api/documents").json()["total"] == 0


def test_a_file_at_the_size_limit_is_accepted(client, bucket, monkeypatch):
    monkeypatch.setattr(documents_router, "MAX_FILE_BYTES", len(TEXT))
    assert upload(client, bucket, [("exact.txt", TEXT)]).json()["parsed"] == 1


def test_too_many_files_in_one_request(client, bucket, monkeypatch):
    monkeypatch.setattr(documents_router, "MAX_FILES_PER_REQUEST", 3)
    response = upload(client, bucket, [(f"f{i}.txt", TEXT + bytes([i])) for i in range(4)])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "too_many_files"


def test_gzipped_upload_is_stored_under_its_real_name(client, bucket):
    upload(client, bucket, [("corpus.txt.gz", gzip.compress(TEXT))])
    document = client.get("/api/documents").json()["items"][0]
    assert document["filename"] == "corpus.txt"
    assert document["status"] == "parsed"
