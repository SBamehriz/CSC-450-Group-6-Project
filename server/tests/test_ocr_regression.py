import io
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from forge.converter import convert_bytes_with_metadata, convert_file
from forge.ingest import ParseError, parse_image_with_metadata
from forge.ocr import (
    get_ocr_engine_name,
    is_ocr_available,
    ocr_document,
    ocr_image,
    ocr_pdf,
)


def _make_text_image(text: str, size: tuple[int, int] = (320, 100), fmt: str = "PNG") -> bytes:
    """Helper to generate a valid RGB test image with crisp black-on-white text."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_ocr_engine_availability_and_name():
    """Verify an OCR engine is available and provides a valid engine name."""
    assert is_ocr_available() is True
    name = get_ocr_engine_name()
    assert isinstance(name, str)
    assert len(name) > 0
    assert name != "None"


def test_ocr_valid_png_with_readable_text(tmp_path: Path):
    """Verify readable PNG image extracts expected terms and consistent metadata."""
    png_bytes = _make_text_image("Invoice Summary Total 2026", size=(320, 100), fmt="PNG")
    png_file = tmp_path / "invoice_summary.png"
    png_file.write_bytes(png_bytes)

    # 1. Direct ocr_document dispatcher
    dispatcher_text = ocr_document(png_bytes, filename="invoice_summary.png")
    assert any(term in dispatcher_text for term in ("Invoice", "Summary", "Total", "2026"))

    # 2. File conversion with extraction metadata
    record = convert_file(png_file, include_metadata=True)
    assert record["filename"] == "invoice_summary.png"
    assert any(term in record["text"] for term in ("Invoice", "Summary", "Total", "2026"))

    meta = record["metadata"]
    assert meta["source_format"] == "png"
    assert meta["ocr_applied"] is True
    assert meta["engine"] == get_ocr_engine_name()
    assert meta["width"] == 320
    assert meta["height"] == 100
    assert meta["char_count"] > 0
    assert meta["word_count"] > 0


def test_ocr_valid_jpg_with_readable_text(tmp_path: Path):
    """Verify readable JPG/JPEG image extracts expected terms and consistent metadata."""
    jpg_bytes = _make_text_image("Shipping Order Details 2026", size=(320, 100), fmt="JPEG")
    jpg_file = tmp_path / "shipping_order.jpg"
    jpg_file.write_bytes(jpg_bytes)

    # 1. Direct ocr_image
    extracted = ocr_image(jpg_file)
    assert any(term in extracted for term in ("Shipping", "Order", "Details", "2026"))

    # 2. Byte conversion with metadata
    text, meta = convert_bytes_with_metadata("shipping_order.jpg", jpg_bytes)
    assert any(term in text for term in ("Shipping", "Order", "Details", "2026"))
    assert meta["source_format"] in ("jpg", "jpeg")
    assert meta["ocr_applied"] is True
    assert meta["engine"] == get_ocr_engine_name()
    assert meta["width"] == 320
    assert meta["height"] == 100


def test_ocr_scanned_pdf(tmp_path: Path):
    """Verify OCR on a raster-based scanned PDF document."""
    # Generate raster PDF via Pillow
    img = Image.new("RGB", (360, 120), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "Confidential Agreement Contract 2026", fill="black")
    pdf_path = tmp_path / "scanned_contract.pdf"
    img.save(str(pdf_path), format="PDF")

    # Direct PDF OCR
    text = ocr_pdf(pdf_path)
    assert any(term in text for term in ("Confidential", "Agreement", "Contract", "2026"))

    # Via unified ocr_document
    doc_text = ocr_document(pdf_path)
    assert any(term in doc_text for term in ("Confidential", "Agreement", "Contract", "2026"))


def test_ocr_image_with_no_readable_text(tmp_path: Path):
    """Verify clean error handling when an image contains no readable text."""
    blank_img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    blank_img.save(buf, format="PNG")
    blank_bytes = buf.getvalue()

    # parse_image_with_metadata should raise ParseError cleanly
    with pytest.raises(ParseError, match="No text could be extracted"):
        parse_image_with_metadata(blank_bytes, filename="blank.png")


def test_ocr_invalid_damaged_image_input():
    """Verify damaged/corrupted image bytes fail cleanly without crashing."""
    corrupted_png = b"\x89PNG\r\n\x1a\n\x00\x00corrupted_image_payload_junk"
    with pytest.raises(ParseError):
        parse_image_with_metadata(corrupted_png, filename="corrupt.png")

    corrupted_jpg = b"\xff\xd8\xffnot_a_valid_jpeg_at_all"
    with pytest.raises(ParseError):
        parse_image_with_metadata(corrupted_jpg, filename="corrupt.jpg")


@pytest.fixture()
def bucket(client):
    response = client.post("/api/buckets", json={"name": f"ocr-bucket-{uuid.uuid4().hex[:6]}"})
    return response.json()["id"]


def test_ocr_damaged_image_in_upload_batch(client, bucket):
    """Verify damaged images do not crash the upload batch workflow."""
    corrupted_bytes = b"\x89PNG\r\n\x1a\ncorrupted_data_here"
    good_text = b"This is perfectly valid ordinary text that easily passes all threshold checks. " * 8

    res = client.post(
        f"/api/buckets/{bucket}/documents",
        files=[
            ("files", ("good.txt", good_text, "text/plain")),
            ("files", ("damaged.png", corrupted_bytes, "image/png")),
        ],
    )
    assert res.status_code == 201
    body = res.json()
    assert body["parsed"] == 1
    assert body["failed"] == 1

    by_name = {o["filename"]: o for o in body["outcomes"]}
    assert by_name["good.txt"]["status"] == "parsed"
    assert by_name["damaged.png"]["status"] == "failed"
    assert by_name["damaged.png"]["error"] is not None


def test_ocr_blank_image_in_upload_batch(client, bucket):
    """Verify an image with no text records failed status cleanly during upload."""
    blank_img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    blank_img.save(buf, format="PNG")

    res = client.post(
        f"/api/buckets/{bucket}/documents",
        files=[("files", ("blank.png", buf.getvalue(), "image/png"))],
    )
    assert res.status_code == 201
    body = res.json()
    assert body["parsed"] == 0
    assert body["failed"] == 1
    assert "No text could be extracted" in body["outcomes"][0]["error"]
