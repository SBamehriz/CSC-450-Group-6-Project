import gzip
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO

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
            f"No article content found in this HTML — only {len(text)} characters "
            "came out. It may be a nav page, a redirect, or rendered entirely by "
            "JavaScript."
        )
    return text


def parse_pdf(raw: bytes, filename: str = "document.pdf") -> str:
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
        return text

    # Scanned PDF or insufficient text layer: fall back to OCR via converter
    try:
        from forge.converter import convert_bytes

        ocr_text = convert_bytes(filename, raw)
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

    return ocr_text


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
        documents.append(Parsed(text=text, filename=f"{filename}#L{number}"))

    if not documents:
        detail = "; ".join(problems[:3]) if problems else "it is empty"
        raise ParseError(f"No usable lines in this JSONL file — {detail}.")
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
        documents.append(Parsed(text=text, filename=item_name))

    if not documents:
        detail = "; ".join(problems[:3]) if problems else "it is empty"
        raise ParseError(f"No usable records in this JSON file — {detail}.")
    return documents


def parse_with_converter(filename: str, raw: bytes) -> list[Parsed]:
    """Parse document using forge.converter (Docling-based) if applicable."""
    from forge.converter import convert_bytes

    try:
        text = convert_bytes(filename, raw)
    except Exception as problem:
        raise ParseError(f"Converter failed on {filename!r}: {problem}") from problem

    if not text.strip():
        raise ParseError(f"No text could be extracted from {filename!r}.")
    return [Parsed(text=text, filename=filename)]


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
        return [Parsed(text=text, filename=filename)]
    if fmt == "html":
        return [Parsed(text=parse_html(raw), filename=filename)]
    if fmt == "pdf":
        return [Parsed(text=parse_pdf(raw, filename=filename), filename=filename)]
    if fmt == "docx" or fmt in IMAGE_EXTENSIONS:
        return parse_with_converter(filename, raw)
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
