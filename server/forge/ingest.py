import gzip
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pypdf
import trafilatura

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp")
SUPPORTED = ("txt", "md", "html", "htm", "pdf", "docx", *IMAGE_EXTENSIONS, "jsonl", "json")

MIN_PDF_CHARS_PER_PAGE = 200
MIN_HTML_CHARS = 200
SHORT_DOC_CHARS = 500
LOW_ALPHA_RATIO = 0.6
LONG_LINE_CHARS = 2000
BOILERPLATE_REPEATS = 5


class ParseError(Exception):
    pass


@dataclass
class Parsed:
    text: str
    filename: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Cleaned:
    text: str
    content_hash: str
    char_count: int
    word_count: int
    quality: dict = field(default_factory=dict)


def split_extension(filename: str) -> tuple[str, bool]:
    # "notes.txt.gz" -> ("txt", True)
    name = filename.lower().strip()
    gzipped = name.endswith(".gz")
    if gzipped:
        name = name[: -len(".gz")]
    _, _, extension = name.rpartition(".")
    return extension, gzipped


def check_supported(filename: str) -> str:
    extension, _ = split_extension(filename)
    if extension not in SUPPORTED:
        raise ParseError(
            f"{filename!r} is a .{extension or '?'} file, which we do not read. "
            f"Supported: {', '.join('.' + e for e in SUPPORTED)}, optionally gzipped."
        )
    # treat htm like html
    return "html" if extension == "htm" else extension


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # latin-1 never fails
        return raw.decode("latin-1")


def parse_html(raw: bytes) -> str:
    text = trafilatura.extract(decode(raw), include_comments=False, include_tables=True)
    text = (text or "").strip()
    if len(text) < MIN_HTML_CHARS:
        raise ParseError(
            f"No article content found in this HTML. Only {len(text)} characters "
            "came out. It may be a nav page, a redirect, or rendered entirely by "
            "JavaScript."
        )
    return text


def parse_pdf_with_metadata(raw: bytes, filename: str = "document.pdf") -> tuple[str, dict[str, Any]]:
    parse_err = None
    pages = []
    try:
        reader = pypdf.PdfReader(BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as problem:
        parse_err = problem

    if not parse_err and not pages:
        raise ParseError("This PDF has no pages.")

    text = "\n\n".join(pages)
    # If text layer has sufficient text, return directly
    if pages and len(text.strip()) >= MIN_PDF_CHARS_PER_PAGE * len(pages):
        metadata = {
            "source_format": "pdf",
            "engine": "pypdf",
            "ocr_applied": False,
            "page_count": len(pages),
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len([line for line in text.splitlines() if line.strip()]),
        }
        return text, metadata

    # Scanned PDF or insufficient text layer: fall back to OCR
    try:
        from forge.ocr import get_ocr_engine_name, ocr_pdf

        ocr_text = ocr_pdf(raw, filename=filename)
    except Exception as problem:
        if parse_err is not None:
            raise ParseError(f"This PDF could not be read ({parse_err}).") from parse_err
        raise ParseError(
            f"This looks like a scanned PDF and OCR conversion failed ({problem})."
        ) from problem

    if not ocr_text.strip():
        if parse_err is not None:
            raise ParseError(f"This PDF could not be read ({parse_err}).") from parse_err
        char_count = len(text.strip())
        page_info = f"across {len(pages)} page(s)" if pages else ""
        raise ParseError(
            f"This looks like a scanned PDF — only {char_count} characters of "
            f"text {page_info}. OCR was unable to extract readable text."
        )

    metadata = {
        "source_format": "pdf",
        "engine": get_ocr_engine_name(),
        "ocr_applied": True,
        "page_count": len(pages),
        "char_count": len(ocr_text),
        "word_count": len(ocr_text.split()),
        "line_count": len([line for line in ocr_text.splitlines() if line.strip()]),
    }
    return ocr_text, metadata


def parse_pdf(raw: bytes, filename: str = "document.pdf") -> str:
    text, _ = parse_pdf_with_metadata(raw, filename=filename)
    return text


def parse_jsonl(raw: bytes, filename: str) -> list[Parsed]:
    documents: list[Parsed] = []
    problems: list[str] = []

    for number, line in enumerate(decode(raw).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {number} is not valid JSON")
            continue
        if not isinstance(record, dict) or "text" not in record:
            problems.append(f"line {number} has no 'text' field")
            continue
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            problems.append(f"line {number} has an empty 'text'")
            continue
        meta = record.get("metadata")
        if not isinstance(meta, dict):
            meta = {
                "source_format": "jsonl",
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": len([line for line in text.splitlines() if line.strip()]),
            }
        documents.append(Parsed(text=text, filename=f"{filename}#L{number}", metadata=meta))

    if not documents:
        detail = "; ".join(problems[:3]) if problems else "it is empty"
        raise ParseError(f"No usable lines in this JSONL file, {detail}.")
    return documents


def parse_json(raw: bytes, filename: str) -> list[Parsed]:
    try:
        data = json.loads(decode(raw))
    except json.JSONDecodeError as err:
        raise ParseError(f"This JSON file could not be decoded ({err}).") from err

    records = data if isinstance(data, list) else [data]
    documents: list[Parsed] = []
    problems: list[str] = []

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or "text" not in record:
            problems.append(f"record {index} has no 'text' field")
            continue
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            problems.append(f"record {index} has an empty 'text'")
            continue
        rec_filename = record.get("filename")
        item_name = (
            rec_filename
            if isinstance(rec_filename, str) and rec_filename.strip()
            else f"{filename}#{index}"
        )
        meta = record.get("metadata")
        if not isinstance(meta, dict):
            meta = {
                "source_format": "json",
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": len([line for line in text.splitlines() if line.strip()]),
            }
        documents.append(Parsed(text=text, filename=item_name, metadata=meta))

    if not documents:
        detail = "; ".join(problems[:3]) if problems else "it is empty"
        raise ParseError(f"No usable records in this JSON file, {detail}.")
    return documents


def parse_docx_with_metadata(raw: bytes, filename: str = "document.docx") -> tuple[str, dict[str, Any]]:
    """Parse DOCX file, returning extracted text and metadata with fallback to Docling."""
    try:
        text, meta = extract_docx_with_metadata(raw)
    except ParseError:
        raise
    except Exception as problem:
        try:
            from forge.converter import convert_bytes

            text = convert_bytes(filename, raw)
            return text, {
                "source_format": "docx",
                "engine": "docling",
                "ocr_applied": False,
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": len([line for line in text.splitlines() if line.strip()]),
            }
        except Exception:
            raise ParseError(f"This DOCX file could not be read ({problem}).") from problem

    if not text.strip():
        try:
            from forge.converter import convert_bytes

            docling_text = convert_bytes(filename, raw)
            if docling_text.strip():
                return docling_text, {
                    "source_format": "docx",
                    "engine": "docling",
                    "ocr_applied": False,
                    "char_count": len(docling_text),
                    "word_count": len(docling_text.split()),
                    "line_count": len([line for line in docling_text.splitlines() if line.strip()]),
                }
        except Exception:
            pass
        raise ParseError(f"No text could be extracted from {filename!r}.")

    return text, meta


def parse_docx(raw: bytes, filename: str = "document.docx") -> str:
    """Parse DOCX file, extracting paragraphs and tables with fallback to Docling."""
    text, _ = parse_docx_with_metadata(raw, filename=filename)
    return text


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{W_NS}}}"


def _heading_prefix(style_val: str) -> str:
    cleaned = style_val.lower().replace(" ", "").replace("_", "").replace("-", "")
    if cleaned in ("heading1", "title"):
        return "# "
    if cleaned in ("heading2", "subtitle"):
        return "## "
    if cleaned == "heading3":
        return "### "
    if cleaned == "heading4":
        return "#### "
    if cleaned == "heading5":
        return "##### "
    if cleaned == "heading6":
        return "###### "
    return ""


def _extract_paragraph_text(p_elem: ET.Element) -> str:
    parts: list[str] = []
    for elem in p_elem.iter():
        tag = elem.tag
        if tag == f"{_W}t" or tag.endswith("}t"):
            if elem.text:
                parts.append(elem.text)
        elif tag == f"{_W}tab" or tag.endswith("}tab"):
            parts.append("\t")
        elif tag in (f"{_W}br", f"{_W}cr") or tag.endswith("}br") or tag.endswith("}cr"):
            parts.append("\n")
    return "".join(parts).strip()


def _extract_cell_text(tc_elem: ET.Element) -> str:
    cell_parts: list[str] = []
    for child in tc_elem:
        if child.tag == f"{_W}p" or child.tag.endswith("}p"):
            p_text = _extract_paragraph_text(child)
            if p_text:
                cell_parts.append(p_text)
    return " ".join(cell_parts).replace("\r\n", " ").replace("\n", " ").replace("|", "\\|").strip()


def _format_table(tbl_elem: ET.Element) -> str:
    rows: list[list[str]] = []
    for tr in tbl_elem:
        if tr.tag == f"{_W}tr" or tr.tag.endswith("}tr"):
            row: list[str] = []
            for tc in tr:
                if tc.tag == f"{_W}tc" or tc.tag.endswith("}tc"):
                    row.append(_extract_cell_text(tc))
            if any(cell.strip() for cell in row):
                rows.append(row)

    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    if max_cols == 0:
        return ""

    padded = [r + [""] * (max_cols - len(r)) for r in rows]
    header = "| " + " | ".join(padded[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(max_cols)) + " |"

    if len(padded) == 1:
        return f"{header}\n{separator}"

    body_lines = ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join([header, separator] + body_lines)


def extract_docx_with_metadata(raw: bytes) -> tuple[str, dict[str, Any]]:
    """Extract paragraphs and tables from raw DOCX bytes with structure metadata."""
    try:
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            if "word/document.xml" not in zf.namelist():
                raise ParseError("Invalid DOCX file: word/document.xml not found.")
            xml_bytes = zf.read("word/document.xml")
    except zipfile.BadZipFile as err:
        raise ParseError(f"This is not a valid DOCX file ({err}).") from err
    except ParseError:
        raise
    except Exception as err:
        raise ParseError(f"Failed to unpack DOCX archive ({err}).") from err

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as err:
        raise ParseError(f"Could not parse DOCX document XML ({err}).") from err

    body = root.find(f"{_W}body")
    if body is None:
        body = root.find("body")
    if body is None:
        raise ParseError("Invalid DOCX: document body not found.")

    blocks: list[str] = []
    paragraph_count = 0
    heading_count = 0
    table_count = 0

    for child in body:
        tag = child.tag
        if tag == f"{_W}p" or tag.endswith("}p"):
            text = _extract_paragraph_text(child)
            if text:
                paragraph_count += 1
                p_pr = child.find(f"{_W}pPr")
                prefix = ""
                if p_pr is not None:
                    style = p_pr.find(f"{_W}pStyle")
                    if style is not None:
                        val = style.get(f"{_W}val") or style.get("val") or ""
                        prefix = _heading_prefix(val)
                    if prefix.startswith("#"):
                        heading_count += 1
                    if not prefix and p_pr.find(f"{_W}numPr") is not None:
                        prefix = "- "
                blocks.append(f"{prefix}{text}")
        elif tag == f"{_W}tbl" or tag.endswith("}tbl"):
            tbl_md = _format_table(child)
            if tbl_md:
                table_count += 1
                blocks.append(tbl_md)

    text = "\n\n".join(blocks).strip()
    metadata = {
        "source_format": "docx",
        "engine": "openxml",
        "ocr_applied": False,
        "paragraph_count": paragraph_count,
        "heading_count": heading_count,
        "table_count": table_count,
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len([line for line in text.splitlines() if line.strip()]),
    }
    return text, metadata


def extract_docx_paragraphs_and_tables(raw: bytes) -> str:
    """Extract paragraphs and tables from raw DOCX bytes in document order."""
    text, _ = extract_docx_with_metadata(raw)
    return text


def parse_image_with_metadata(raw: bytes, filename: str = "image.png") -> tuple[str, dict[str, Any]]:
    """Parse raster image (.png, .jpg, .jpeg, etc.) using OCR and return text with metadata."""
    from PIL import Image

    from forge.ocr import get_ocr_engine_name, ocr_image

    img_info: dict[str, Any] = {}
    try:
        with Image.open(BytesIO(raw)) as img:
            img_info = {
                "width": img.width,
                "height": img.height,
                "image_mode": img.mode,
                "image_format": img.format or Path(filename).suffix.lstrip(".").upper(),
            }
    except Exception:
        pass

    try:
        text = ocr_image(raw, filename=filename)
    except Exception as problem:
        raise ParseError(f"OCR failed on {filename!r}: {problem}") from problem

    if not text.strip():
        raise ParseError(f"No text could be extracted from {filename!r} via OCR.")

    ext, _ = split_extension(filename)
    metadata = {
        "source_format": ext or "image",
        "engine": get_ocr_engine_name(),
        "ocr_applied": True,
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len([line for line in text.splitlines() if line.strip()]),
        **img_info,
    }
    return text, metadata


def parse_image(raw: bytes, filename: str = "image.png") -> str:
    """Parse raster image (.png, .jpg, .jpeg, etc.) using OCR."""
    text, _ = parse_image_with_metadata(raw, filename=filename)
    return text


def parse_with_converter(filename: str, raw: bytes) -> list[Parsed]:
    """Parse document using forge.converter (Docling-based) if applicable."""
    from forge.converter import convert_bytes

    try:
        text = convert_bytes(filename, raw)
    except Exception as problem:
        raise ParseError(f"Converter failed on {filename!r}: {problem}") from problem

    if not text.strip():
        raise ParseError(f"No text could be extracted from {filename!r}.")
    ext, _ = split_extension(filename)
    meta = {
        "source_format": ext,
        "engine": "docling",
        "ocr_applied": False,
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len([line for line in text.splitlines() if line.strip()]),
    }
    return [Parsed(text=text, filename=filename, metadata=meta)]


def parse(filename: str, raw: bytes) -> list[Parsed]:
    fmt = check_supported(filename)
    _, gzipped = split_extension(filename)

    if gzipped:
        try:
            raw = gzip.decompress(raw)
        except OSError as problem:
            raise ParseError(f"This .gz file could not be unpacked ({problem}).") from problem
        filename = filename[: -len(".gz")]

    if fmt in ("txt", "md"):
        text = decode(raw)
        if not text.strip():
            raise ParseError("This file is empty.")
        meta = {
            "source_format": fmt,
            "engine": "native",
            "ocr_applied": False,
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len([line for line in text.splitlines() if line.strip()]),
        }
        return [Parsed(text=text, filename=filename, metadata=meta)]
    if fmt == "html":
        text = parse_html(raw)
        meta = {
            "source_format": "html",
            "engine": "trafilatura",
            "ocr_applied": False,
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len([line for line in text.splitlines() if line.strip()]),
        }
        return [Parsed(text=text, filename=filename, metadata=meta)]
    if fmt == "pdf":
        text, meta = parse_pdf_with_metadata(raw, filename=filename)
        return [Parsed(text=text, filename=filename, metadata=meta)]
    if fmt == "docx":
        text, meta = parse_docx_with_metadata(raw, filename=filename)
        return [Parsed(text=text, filename=filename, metadata=meta)]
    if fmt in IMAGE_EXTENSIONS:
        text, meta = parse_image_with_metadata(raw, filename=filename)
        return [Parsed(text=text, filename=filename, metadata=meta)]
    if fmt == "json":
        return parse_json(raw, filename)
    return parse_jsonl(raw, filename)


# keep \n and \t, drop other control chars
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BLANK_RUN = re.compile(r"\n{3,}")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANK_RUN.sub("\n\n", text).strip()


def drop_repeated_lines(text: str) -> str:
    # a long repeat is usually real text
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            counts[stripped] = counts.get(stripped, 0) + 1

    junk = {
        line for line, count in counts.items() if count >= BOILERPLATE_REPEATS and len(line) <= 80
    }
    if not junk:
        return text
    kept = "\n".join(line for line in lines if line.strip() not in junk)
    return _BLANK_RUN.sub("\n\n", kept).strip()


def measure(text: str) -> dict:
    lines = [line for line in text.split("\n") if line.strip()]
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    non_ascii = sum(1 for c in text if ord(c) > 127)
    total = len(text) or 1
    return {
        "alpha_ratio": round(letters / total, 3),
        "digit_ratio": round(digits / total, 3),
        "non_ascii_ratio": round(non_ascii / total, 3),
        "mean_line_len": round(sum(len(line) for line in lines) / len(lines), 1) if lines else 0.0,
        "line_count": len(lines),
    }


def flag(text: str, stats: dict, encoding_suspect: bool = False) -> list[str]:
    flags = []
    if len(text) < SHORT_DOC_CHARS:
        flags.append("too_short")
    if stats["alpha_ratio"] < LOW_ALPHA_RATIO:
        flags.append("low_alpha")
    if stats["mean_line_len"] > LONG_LINE_CHARS:
        flags.append("long_lines")
    if encoding_suspect:
        flags.append("encoding_suspect")
    return flags


def clean(text: str, mode: str = "standard", encoding_suspect: bool = False) -> Cleaned:
    if mode == "standard":
        text = drop_repeated_lines(normalise(text))
    elif mode != "none":
        raise ValueError(f"unknown cleaning mode {mode!r}")

    stats = measure(text)
    return Cleaned(
        text=text,
        # dedup compares the cleaned text
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        char_count=len(text),
        word_count=len(text.split()),
        quality={**stats, "flags": flag(text, stats, encoding_suspect)},
    )


def looks_mis_decoded(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True
