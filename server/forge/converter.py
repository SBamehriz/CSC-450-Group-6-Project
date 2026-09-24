import argparse
import json
import os
import sys
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

# Suppress harmless Windows symlink warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from forge.ingest import (
    Cleaned,
    Parsed,
    clean,
    extract_docx_paragraphs_and_tables,
    extract_docx_with_metadata,
    parse_docx,
    parse_docx_with_metadata,
    parse_image_with_metadata,
    parse_pdf_with_metadata,
)
from forge.ocr import (
    get_ocr_engine_name,
    is_ocr_available,
    ocr_document,
    ocr_image,
    ocr_pdf,
)

_converter = None
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")
SUPPORTED_CONVERTER_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    *SUPPORTED_IMAGE_EXTENSIONS,
)


def get_converter():
    """Lazily load and return the Docling DocumentConverter singleton."""
    global _converter
    if _converter is None:
        try:
            from docling.document_converter import DocumentConverter

            _converter = DocumentConverter()
        except ImportError as err:
            raise RuntimeError(
                "The 'docling' package is required for document conversion. "
                "Install it via 'pip install docling' or 'pip install -e .[converter]'."
            ) from err
    return _converter


def convert_file(
    file_path: Path | str,
    clean_text: bool = False,
    clean_mode: str = "standard",
    force_ocr: bool = False,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Convert a single file (.pdf, .docx, image, .txt, .md) to a record dict:

    {'filename': <name>, 'text': <extracted_content>, 'metadata': <extraction_meta>}
    If clean_text=True, runs the extracted text through the cleaning pipeline and returns:
    {'filename': <name>, 'text': <cleaned_content>, 'content_hash': <hash>,
     'char_count': <chars>, 'word_count': <words>, 'quality': <quality_dict>, 'metadata': <meta>}
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    metadata: dict[str, Any] = {}

    if ext in (".txt", ".md"):
        # utf-8-sig transparently handles standard UTF-8 and UTF-8 with BOM
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        metadata = {
            "source_format": ext.lstrip("."),
            "engine": "native",
            "ocr_applied": False,
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len([line for line in text.splitlines() if line.strip()]),
        }
    elif ext == ".docx":
        text, metadata = parse_docx_with_metadata(path.read_bytes(), filename=path.name)
    elif ext == ".pdf":
        raw = path.read_bytes()
        if force_ocr:
            text = ocr_pdf(raw, filename=path.name)
            metadata = {
                "source_format": "pdf",
                "engine": get_ocr_engine_name(),
                "ocr_applied": True,
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": len([line for line in text.splitlines() if line.strip()]),
            }
        else:
            try:
                text, metadata = parse_pdf_with_metadata(raw, filename=path.name)
            except Exception:
                text = ocr_pdf(raw, filename=path.name)
                metadata = {
                    "source_format": "pdf",
                    "engine": get_ocr_engine_name(),
                    "ocr_applied": True,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "line_count": len([line for line in text.splitlines() if line.strip()]),
                }
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        text, metadata = parse_image_with_metadata(path.read_bytes(), filename=path.name)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported extensions: {', '.join(SUPPORTED_CONVERTER_EXTENSIONS)}"
        )

    res: dict[str, Any]
    if clean_text:
        cleaned = clean(text, mode=clean_mode)
        res = {
            "filename": path.name,
            "text": cleaned.text,
            "content_hash": cleaned.content_hash,
            "char_count": cleaned.char_count,
            "word_count": cleaned.word_count,
            "quality": cleaned.quality,
        }
    else:
        res = {"filename": path.name, "text": text}

    if include_metadata:
        res["metadata"] = metadata

    return res


def convert_bytes(filename: str, raw: bytes, force_ocr: bool = False) -> str:
    """Extract markdown/text from raw bytes for .pdf, .docx, images, .txt, or .md."""
    text, _ = convert_bytes_with_metadata(filename, raw, force_ocr=force_ocr)
    return text


def convert_bytes_with_metadata(
    filename: str, raw: bytes, force_ocr: bool = False
) -> tuple[str, dict[str, Any]]:
    """Extract markdown/text and extraction metadata from raw bytes."""
    ext = Path(filename).suffix.lower()
    if ext in (".txt", ".md"):
        text = raw.decode("utf-8-sig", errors="ignore")
        metadata = {
            "source_format": ext.lstrip("."),
            "engine": "native",
            "ocr_applied": False,
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len([line for line in text.splitlines() if line.strip()]),
        }
        return text, metadata
    elif ext == ".docx":
        return parse_docx_with_metadata(raw, filename=filename)
    elif ext == ".pdf":
        if force_ocr:
            text = ocr_pdf(raw, filename=filename)
            metadata = {
                "source_format": "pdf",
                "engine": get_ocr_engine_name(),
                "ocr_applied": True,
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": len([line for line in text.splitlines() if line.strip()]),
            }
            return text, metadata
        try:
            return parse_pdf_with_metadata(raw, filename=filename)
        except Exception:
            text = ocr_pdf(raw, filename=filename)
            metadata = {
                "source_format": "pdf",
                "engine": get_ocr_engine_name(),
                "ocr_applied": True,
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": len([line for line in text.splitlines() if line.strip()]),
            }
            return text, metadata
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        return parse_image_with_metadata(raw, filename=filename)
    else:
        raise ValueError(
            f"Unsupported file format for conversion: {ext}. "
            f"Supported extensions: {', '.join(SUPPORTED_CONVERTER_EXTENSIONS)}"
        )


def convert_to_parsed(file_path: Path | str) -> Parsed:
    """Convert a file into a Parsed dataclass instance suitable for the ingest pipeline."""
    record = convert_file(file_path, include_metadata=True)
    return Parsed(
        text=record["text"],
        filename=record["filename"],
        metadata=record.get("metadata", {}),
    )


def convert_to_cleaned(file_path: Path | str, mode: str = "standard") -> Cleaned:
    """Convert a single file and process it directly through Forge's cleaning pipeline."""
    record = convert_file(file_path)
    return clean(record["text"], mode=mode)


def convert_directory(
    input_dir: Path | str,
    output_file: Path | str | None = None,
    extensions: Sequence[str] = SUPPORTED_CONVERTER_EXTENSIONS,
    clean_text: bool = False,
    clean_mode: str = "standard",
    force_ocr: bool = False,
    include_metadata: bool = True,
) -> list[dict[str, Any]]:
    """Convert all matching files in input_dir into a list of record dicts,

    and optionally save them to output_file in JSON format.
    If clean_text=True, runs all extracted texts through the cleaning pipeline.
    If force_ocr=True, forces OCR extraction on scanned PDFs and images.
    If include_metadata=True, attaches extraction metadata to each record.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    records: list[dict[str, Any]] = []

    for file_path in sorted(input_path.glob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in allowed:
            continue
        try:
            record = convert_file(
                file_path,
                clean_text=clean_text,
                clean_mode=clean_mode,
                force_ocr=force_ocr,
                include_metadata=include_metadata,
            )
            records.append(record)
            print(f"Converted: {file_path.name}")
        except Exception as e:
            print(f"Failed on {file_path.name}: {e}", file=sys.stderr)

    if output_file is not None:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    return records


def _resolve_bucket_id(session: Session, bucket: uuid.UUID | str) -> uuid.UUID:
    """Resolve a bucket name or string UUID to a uuid.UUID instance."""
    if isinstance(bucket, uuid.UUID):
        return bucket
    try:
        return uuid.UUID(bucket)
    except ValueError:
        from sqlalchemy import select

        from forge.models import Bucket

        matched = session.execute(select(Bucket).where(Bucket.name == bucket)).scalar_one_or_none()
        if matched is None:
            raise ValueError(f"No bucket found matching name or UUID: {bucket!r}")
        return matched.id


def ingest_converted(
    session: Session,
    bucket_id: uuid.UUID,
    filename: str,
    text: str,
    source_note: str = "",
    raw_bytes: bytes | None = None,
    clean_mode: str = "standard",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Ingest a converted document directly into a bucket in Forge.

    Performs normalization, cleaning, twin deduplication against existing bucket content,
    raw/cleaned disk storage, and database persistence.
    """
    from sqlalchemy import select

    from forge.documents import _stored_format, _write
    from forge.models import Document

    raw = raw_bytes if raw_bytes is not None else text.encode("utf-8")
    raw_uri = _write(f"raw/{uuid.uuid4()}", raw)

    cleaned = clean(text, mode=clean_mode)
    doc_quality = dict(cleaned.quality)
    if metadata:
        doc_quality["extraction"] = metadata

    twin = session.execute(
        select(Document).where(
            Document.bucket_id == bucket_id,
            Document.content_hash == cleaned.content_hash,
        )
    ).scalar_one_or_none()

    source_fmt = _stored_format(filename)

    if twin is not None:
        message = f"Duplicate of {twin.filename}. Identical text after cleaning."
        document = Document(
            bucket_id=bucket_id,
            filename=filename,
            source_format=source_fmt,
            source_note=source_note,
            raw_uri=raw_uri,
            status="failed",
            error=message,
            quality=doc_quality,
        )
        session.add(document)
        session.flush()
        return document

    document = Document(
        bucket_id=bucket_id,
        filename=filename,
        source_format=source_fmt,
        source_note=source_note,
        raw_uri=raw_uri,
        status="parsed",
        content_hash=cleaned.content_hash,
        char_count=cleaned.char_count,
        word_count=cleaned.word_count,
        quality=doc_quality,
    )
    session.add(document)
    session.flush()
    document.text_uri = _write(f"text/{document.id}.txt", cleaned.text.encode("utf-8"))
    return document


def convert_and_upload(
    file_path: Path | str,
    bucket: uuid.UUID | str,
    session: Session | None = None,
    source_note: str = "",
    clean_mode: str = "standard",
    force_ocr: bool = False,
) -> Any:
    """Convert a file from disk and upload/ingest it directly into a Forge bucket."""
    path = Path(file_path)
    raw_bytes = path.read_bytes()
    record = convert_file(path, force_ocr=force_ocr, include_metadata=True)

    def _execute(s: Session):
        bucket_id = _resolve_bucket_id(s, bucket)
        return ingest_converted(
            session=s,
            bucket_id=bucket_id,
            filename=path.name,
            text=record["text"],
            source_note=source_note,
            raw_bytes=raw_bytes,
            clean_mode=clean_mode,
            metadata=record.get("metadata"),
        )

    if session is not None:
        doc = _execute(session)
        session.flush()
        return doc

    from forge.db import _SessionLocal, get_engine

    get_engine()
    assert _SessionLocal is not None
    with _SessionLocal() as s:
        doc = _execute(s)
        s.commit()
        return doc


def convert_directory_and_upload(
    input_dir: Path | str,
    bucket: uuid.UUID | str,
    session: Session | None = None,
    source_note: str = "",
    extensions: Sequence[str] = SUPPORTED_CONVERTER_EXTENSIONS,
    clean_mode: str = "standard",
    force_ocr: bool = False,
) -> list[Any]:
    """Convert a directory of documents and ingest them directly into a Forge bucket."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}

    def _execute_batch(s: Session) -> list[Any]:
        bucket_id = _resolve_bucket_id(s, bucket)
        docs = []
        for file_path in sorted(input_path.glob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in allowed:
                continue
            try:
                raw_bytes = file_path.read_bytes()
                record = convert_file(file_path, force_ocr=force_ocr, include_metadata=True)
                doc = ingest_converted(
                    session=s,
                    bucket_id=bucket_id,
                    filename=file_path.name,
                    text=record["text"],
                    source_note=source_note,
                    raw_bytes=raw_bytes,
                    clean_mode=clean_mode,
                    metadata=record.get("metadata"),
                )
                docs.append(doc)
                print(f"Ingested {file_path.name} -> {doc.status}")
            except Exception as e:
                print(f"Failed on {file_path.name}: {e}", file=sys.stderr)
        return docs

    if session is not None:
        docs = _execute_batch(session)
        session.flush()
        return docs

    from forge.db import _SessionLocal, get_engine

    get_engine()
    assert _SessionLocal is not None
    with _SessionLocal() as s:
        docs = _execute_batch(s)
        s.commit()
        return docs


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF, DOCX, image, and text files to JSON using Docling."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="input_files",
        help="Input directory containing files (default: input_files)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.json",
        help="Output JSON file path (default: output.json)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run extracted text through Forge cleaning and quality validation pipeline",
    )
    parser.add_argument(
        "--clean-mode",
        choices=["standard", "none"],
        default="standard",
        help="Cleaning mode: standard (boilerplate/normalise) or none (default: standard)",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Force OCR processing on scanned PDFs and raster images",
    )
    parser.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include extraction metadata in output records (default: enabled).",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Target Forge bucket name or UUID to upload and ingest converted files directly",
    )
    parser.add_argument(
        "--source-note",
        default="",
        help="Source note describing provenance of uploaded documents",
    )
    parser.add_argument("files", nargs="*", help="Optional specific file(s) to convert")

    args = parser.parse_args()

    if args.bucket:
        if args.files:
            docs = []
            for file_arg in args.files:
                try:
                    doc = convert_and_upload(
                        file_arg,
                        bucket=args.bucket,
                        source_note=args.source_note,
                        clean_mode=args.clean_mode,
                        force_ocr=args.ocr,
                    )
                    docs.append(doc)
                    print(f"Ingested {Path(file_arg).name} -> {doc.status}")
                except Exception as e:
                    print(f"Failed on {file_arg}: {e}", file=sys.stderr)
            print(f"\nDone. {len(docs)} file(s) uploaded to bucket {args.bucket}")
        else:
            docs = convert_directory_and_upload(
                args.input_dir,
                bucket=args.bucket,
                source_note=args.source_note,
                clean_mode=args.clean_mode,
                force_ocr=args.ocr,
            )
            print(f"\nDone. {len(docs)} file(s) uploaded to bucket {args.bucket}")
        return

    if args.files:
        records = []
        for file_arg in args.files:
            try:
                record = convert_file(
                    file_arg,
                    clean_text=args.clean,
                    clean_mode=args.clean_mode,
                    force_ocr=args.ocr,
                    include_metadata=args.metadata,
                )
                records.append(record)
                print(f"Converted: {Path(file_arg).name}")
            except Exception as e:
                print(f"Failed on {file_arg}: {e}", file=sys.stderr)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"\nDone. {len(records)} file(s) converted -> {args.output}")
        else:
            print(json.dumps(records, indent=2, ensure_ascii=False))
    else:
        records = convert_directory(
            args.input_dir,
            args.output,
            clean_text=args.clean,
            clean_mode=args.clean_mode,
            force_ocr=args.ocr,
            include_metadata=args.metadata,
        )
        print(f"\nDone. {len(records)} file(s) converted -> {args.output}")


if __name__ == "__main__":
    main()

