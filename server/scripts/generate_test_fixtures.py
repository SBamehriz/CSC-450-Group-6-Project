"""Generate test fixtures for docx, images, and pdfs using standard library only."""
import io
import struct
import zlib
import zipfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_docx(out_path: Path):
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>'
    )
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Forge Architecture Overview</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>Forge is a data lab and pre-training workbench designed for tiny Language Models.</w:t></w:r></w:p>'
        '<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Core Components</w:t></w:r></w:p>'
        '<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>Document conversion and layout analysis</w:t></w:r></w:p>'
        '<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>Heuristic text cleaning and quality scoring</w:t></w:r></w:p>'
        '<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Benchmark Results</w:t></w:r></w:p>'
        '<w:tbl>'
        '<w:tr>'
        '<w:tc><w:p><w:r><w:t>Component</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>Throughput</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>Status</w:t></w:r></w:p></w:tc>'
        '</w:tr>'
        '<w:tr>'
        '<w:tc><w:p><w:r><w:t>DOCX Parser</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>120 docs/s</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>Optimal</w:t></w:r></w:p></w:tc>'
        '</w:tr>'
        '<w:tr>'
        '<w:tc><w:p><w:r><w:t>OCR Engine</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>45 pages/s</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>Active</w:t></w:r></w:p></w:tc>'
        '</w:tr>'
        '</w:tbl>'
        '<w:p><w:r><w:t>All pipeline stages operate in streaming fashion.</w:t></w:r></w:p>'
        '</w:body>'
        '</w:document>'
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", doc_xml)

    out_path.write_bytes(buf.getvalue())
    print(f"Created {out_path} ({out_path.stat().st_size} bytes)")


def create_sample_png(out_path: Path):
    # Valid PNG: 100x40 RGB white image with simple IDAT chunk
    width, height = 100, 40
    raw_rows = []
    for _ in range(height):
        # filter byte 0, then 100 * 3 bytes of 255 (white)
        raw_rows.append(b"\x00" + b"\xff\xff\xff" * width)
    raw_data = b"".join(raw_rows)
    compressed = zlib.compress(raw_data)

    png_header = b"\x89PNG\r\n\x1a\n"
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    # IDAT
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed))
    idat_chunk = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc

    # IEND
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND"))
    iend_chunk = struct.pack(">I", 0) + b"IEND" + iend_crc

    png_bytes = png_header + ihdr_chunk + idat_chunk + iend_chunk
    out_path.write_bytes(png_bytes)
    print(f"Created {out_path} ({out_path.stat().st_size} bytes)")


def create_sample_jpg(out_path: Path):
    # Minimal 1x1 valid JFIF JPEG
    jpg_bytes = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48,
        0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00,
        0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0xFF, 0xDA, 0x00, 0x08, 0x01,
        0x01, 0x00, 0x00, 0x3F, 0x00, 0x37, 0xFF, 0xD9
    ])
    out_path.write_bytes(jpg_bytes)
    print(f"Created {out_path} ({out_path.stat().st_size} bytes)")


def _make_pdf(lines: list[str]) -> bytes:
    content = "BT /F1 12 Tf 72 720 Td 14 TL\n"
    for line in lines:
        content += f"({line}) Tj T*\n"
    content += "ET"
    stream = content.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def create_sample_pdf(out_path: Path):
    pdf_bytes = _make_pdf([
        "Forge Document Ingestion and Conversion Fixture",
        "This is page one of the sample PDF document with structured text.",
        "The extractor should parse all sentences with proper spacing and density.",
        "Multiple paragraphs verify layout analysis and char count calculations.",
    ] * 3)
    out_path.write_bytes(pdf_bytes)
    print(f"Created {out_path} ({out_path.stat().st_size} bytes)")


def create_scanned_sample_pdf(out_path: Path):
    # Minimal scan-like PDF (low character density to trigger OCR fallback)
    pdf_bytes = _make_pdf(["12"])
    out_path.write_bytes(pdf_bytes)
    print(f"Created {out_path} ({out_path.stat().st_size} bytes)")


def main():
    create_sample_docx(FIXTURES_DIR / "sample.docx")
    create_sample_png(FIXTURES_DIR / "sample.png")
    create_sample_jpg(FIXTURES_DIR / "sample.jpg")
    create_sample_pdf(FIXTURES_DIR / "sample.pdf")
    create_scanned_sample_pdf(FIXTURES_DIR / "scanned_sample.pdf")
    print("All fixtures generated successfully.")


if __name__ == "__main__":
    main()
