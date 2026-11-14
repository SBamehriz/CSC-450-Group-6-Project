import io
import json
import sys
from pathlib import Path

# Ensure local forge package takes precedence
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from forge.converter import convert_bytes, convert_bytes_with_metadata
from forge.ingest import ParseError, parse, parse_json
from forge.ocr import (
    get_ocr_engine_name,
    is_ocr_available,
    ocr_document,
    ocr_image,
    ocr_pdf,
)

FIXTURES = Path(__file__).parent / "fixtures"
requires_ocr = pytest.mark.skipif(
    not is_ocr_available(), reason="Install the ocr extra to test OCR"
)


def test_convert_bytes_txt():
    text = convert_bytes("sample.txt", b"In-memory raw text bytes")
    assert text == "In-memory raw text bytes"


def test_convert_bytes_unsupported():
    with pytest.raises(ValueError, match="Unsupported file format"):
        convert_bytes("data.csv", b"1,2,3")


def test_parse_json_from_converter():
    records = [
        {"filename": "doc1.pdf", "text": "Extracted text from doc1"},
        {"filename": "doc2.docx", "text": "Extracted text from doc2"},
    ]
    raw = json.dumps(records).encode("utf-8")
    parsed_docs = parse_json(raw, "output.json")

    assert len(parsed_docs) == 2
    assert parsed_docs[0].filename == "doc1.pdf"
    assert parsed_docs[0].text == "Extracted text from doc1"
    assert parsed_docs[1].filename == "doc2.docx"
    assert parsed_docs[1].text == "Extracted text from doc2"


def test_parse_json_via_ingest_parse():
    records = [{"filename": "sample.pdf", "text": "Sample text content"}]
    raw = json.dumps(records).encode("utf-8")

    docs = parse("output.json", raw)
    assert len(docs) == 1
    assert docs[0].filename == "sample.pdf"
    assert docs[0].text == "Sample text content"


def test_parse_json_errors():
    with pytest.raises(ParseError, match="could not be decoded"):
        parse_json(b"invalid-json{", "broken.json")

    with pytest.raises(ParseError, match="No usable records"):
        parse_json(json.dumps([{"no_text": 123}]).encode("utf-8"), "empty.json")


@requires_ocr
def test_convert_bytes_image_with_ocr():
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (300, 100), color="white")
    d = ImageDraw.Draw(img)
    d.text((10, 30), "Docling In-Memory OCR", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    text = convert_bytes("test_mem.png", buf.getvalue())
    assert any(term in text for term in ("Docling", "Memory", "OCR"))


def test_convert_bytes_docx_table():
    import io

    import docx

    doc = docx.Document()
    doc.add_heading("In-Memory Docx", level=1)
    tbl = doc.add_table(rows=2, cols=2)
    tbl.cell(0, 0).text = "A"
    tbl.cell(0, 1).text = "B"
    tbl.cell(1, 0).text = "1"
    tbl.cell(1, 1).text = "2"
    buf = io.BytesIO()
    doc.save(buf)

    text = convert_bytes("memory.docx", buf.getvalue())
    assert "# In-Memory Docx" in text
    assert "| A | B |" in text
    assert "| 1 | 2 |" in text


def test_ocr_availability():
    # If test suite runs, check engine name representation
    name = get_ocr_engine_name()
    assert isinstance(name, str)
    assert len(name) > 0


@requires_ocr
def test_ocr_jpg_and_jpeg_images(tmp_path: Path):
    from PIL import Image, ImageDraw

    # Test JPG
    img_jpg = Image.new("RGB", (320, 100), color="white")
    d1 = ImageDraw.Draw(img_jpg)
    d1.text((10, 30), "Invoice JPG Scan 2026", fill="black")
    jpg_path = tmp_path / "receipt.jpg"
    img_jpg.save(str(jpg_path), format="JPEG")

    text_jpg = ocr_image(jpg_path)
    assert any(term in text_jpg for term in ("Invoice", "Scan", "2026"))


@requires_ocr
def test_ocr_scanned_pdf(tmp_path: Path):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (360, 120), color="white")
    d = ImageDraw.Draw(img)
    d.text((10, 30), "Scanned Legal Deed 2026", fill="black")
    pdf_path = tmp_path / "scanned_deed.pdf"
    img.save(str(pdf_path), format="PDF")

    text = ocr_pdf(pdf_path)
    assert any(term in text for term in ("Scanned", "Legal", "Deed", "2026"))


@requires_ocr
def test_ocr_document_dispatcher(tmp_path: Path):
    from PIL import Image, ImageDraw

    # Test PNG through unified dispatcher
    img = Image.new("RGB", (300, 100), color="white")
    d = ImageDraw.Draw(img)
    d.text((10, 30), "Dispatcher PNG OCR", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    text = ocr_document(buf.getvalue(), filename="test.png")
    assert any(term in text for term in ("Dispatcher", "PNG", "OCR"))


def test_convert_bytes_plain_text_metadata():
    raw = b"Hello, extraction metadata!\nSecond line of text."

    text, meta = convert_bytes_with_metadata("metadata_test.txt", raw)
    assert meta["source_format"] == "txt"
    assert meta["engine"] == "native"
    assert meta["ocr_applied"] is False
    assert meta["char_count"] == len(text)
    assert meta["word_count"] == len(text.split())
    assert meta["line_count"] == 2


def test_convert_bytes_docx_fixture_metadata():
    raw = (FIXTURES / "sample.docx").read_bytes()

    text, meta = convert_bytes_with_metadata("sample.docx", raw)
    assert "# Forge Architecture Overview" in text
    assert "## Core Components" in text
    assert "| Component | Throughput | Status |" in text

    assert meta["source_format"] == "docx"
    assert meta["engine"] == "openxml"
    assert meta["ocr_applied"] is False
    assert meta["table_count"] == 1
    assert meta["heading_count"] >= 3
    assert meta["paragraph_count"] >= 5


def test_convert_bytes_pdf_fixture_metadata():
    raw = (FIXTURES / "sample.pdf").read_bytes()

    text, meta = convert_bytes_with_metadata("sample.pdf", raw)
    assert "Forge Document Ingestion" in text

    assert meta["source_format"] == "pdf"
    assert meta["engine"] == "pypdf"
    assert meta["ocr_applied"] is False
    assert meta["page_count"] >= 1


def test_convert_bytes_png_fixture_metadata(monkeypatch):
    monkeypatch.setattr("forge.ocr.ocr_image", lambda *_args, **_kwargs: "Docling Image OCR")
    raw = (FIXTURES / "sample.png").read_bytes()

    text, meta = convert_bytes_with_metadata("sample.png", raw)
    assert any(term in text for term in ("Docling", "Image", "OCR"))
    assert meta["source_format"] == "png"
    assert meta["ocr_applied"] is True
    assert meta["width"] == 300
    assert meta["height"] == 100


def test_convert_bytes_jpg_fixture_metadata(monkeypatch):
    monkeypatch.setattr("forge.ocr.ocr_image", lambda *_args, **_kwargs: "Invoice JPG Scan 2026")
    raw = (FIXTURES / "sample.jpg").read_bytes()

    text, meta = convert_bytes_with_metadata("sample.jpg", raw)
    assert any(term in text for term in ("Invoice", "Scan", "2026"))
    assert meta["source_format"] in ("jpg", "jpeg")
    assert meta["ocr_applied"] is True
    assert meta["width"] == 320
    assert meta["height"] == 100


def test_convert_bytes_with_metadata_docx():
    docx_fixture = FIXTURES / "sample.docx"
    raw = docx_fixture.read_bytes()

    text, meta = convert_bytes_with_metadata("sample.docx", raw)
    assert "# Forge Architecture Overview" in text
    assert "| Component | Throughput | Status |" in text
    assert meta["source_format"] == "docx"
    assert meta["engine"] == "openxml"
    assert meta["table_count"] == 1
