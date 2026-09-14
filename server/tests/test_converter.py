import json
from pathlib import Path

import pytest

from forge.converter import (
    convert_bytes,
    convert_directory,
    convert_file,
    convert_to_parsed,
)
from forge.ingest import ParseError, parse, parse_json, parse_with_converter

FIXTURES = Path(__file__).parent / "fixtures"
DESKTOP_INPUTS = Path(r"C:\Users\PSK\Desktop\pdf-docx-to-json\input_files")


def test_convert_file_plain_text(tmp_path: Path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello, Docling conversion!", encoding="utf-8")

    record = convert_file(txt_file)
    assert record["filename"] == "test.txt"
    assert "Hello, Docling conversion!" in record["text"]


def test_convert_file_markdown(tmp_path: Path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Title\n\n- item 1\n- item 2", encoding="utf-8")

    record = convert_file(md_file)
    assert record["filename"] == "notes.md"
    assert "# Title" in record["text"]


def test_convert_to_parsed(tmp_path: Path):
    txt_file = tmp_path / "parsed_test.txt"
    txt_file.write_text("Parsed content here", encoding="utf-8")

    parsed = convert_to_parsed(txt_file)
    assert parsed.filename == "parsed_test.txt"
    assert parsed.text == "Parsed content here"


def test_convert_directory(tmp_path: Path):
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    (in_dir / "a.txt").write_text("Alpha text", encoding="utf-8")
    (in_dir / "b.md").write_text("Beta text", encoding="utf-8")
    (in_dir / "c.csv").write_text("Col1,Col2", encoding="utf-8")

    out_json = tmp_path / "output.json"
    records = convert_directory(in_dir, output_file=out_json)

    assert len(records) == 2
    filenames = {r["filename"] for r in records}
    assert filenames == {"a.txt", "b.md"}

    assert out_json.is_file()
    saved_records = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(saved_records) == 2


def test_convert_bytes_txt():
    text = convert_bytes("sample.txt", b"In-memory raw text bytes")
    assert text == "In-memory raw text bytes"


def test_convert_file_unsupported(tmp_path: Path):
    bad = tmp_path / "data.csv"
    bad.write_text("1,2,3", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        convert_file(bad)


def test_convert_file_missing():
    with pytest.raises(FileNotFoundError):
        convert_file("non_existent_file.docx")


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


def test_parse_with_converter_txt():
    docs = parse_with_converter("sample.txt", b"Direct converter parse text")
    assert len(docs) == 1
    assert docs[0].text == "Direct converter parse text"
    assert docs[0].filename == "sample.txt"


@pytest.mark.slow
def test_docling_docx_conversion_if_sample_exists():
    docx_sample = DESKTOP_INPUTS / "sample.docx"
    if not docx_sample.is_file():
        pytest.skip(f"Sample docx not found at {docx_sample}")

    record = convert_file(docx_sample)
    assert record["filename"] == "sample.docx"
    assert len(record["text"]) > 0
    assert "Sample Word Document" in record["text"]
