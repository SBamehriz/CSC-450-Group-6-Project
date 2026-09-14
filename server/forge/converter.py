import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

# Suppress harmless Windows symlink warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from forge.ingest import Parsed, ParseError

_converter = None
SUPPORTED_CONVERTER_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


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


def convert_file(file_path: Path | str) -> dict[str, str]:
    """Convert a single file (.pdf, .docx, .txt, .md) to a record dict:

    {'filename': <name>, 'text': <extracted_content>}
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        # utf-8-sig transparently handles standard UTF-8 and UTF-8 with BOM
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    elif ext in (".pdf", ".docx"):
        converter = get_converter()
        result = converter.convert(str(path))
        text = result.document.export_to_markdown()
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported extensions: {', '.join(SUPPORTED_CONVERTER_EXTENSIONS)}"
        )

    return {"filename": path.name, "text": text}


def convert_bytes(filename: str, raw: bytes) -> str:
    """Extract markdown/text from raw bytes for .pdf, .docx, .txt, or .md."""
    ext = Path(filename).suffix.lower()
    if ext in (".txt", ".md"):
        return raw.decode("utf-8-sig", errors="ignore")
    elif ext in (".pdf", ".docx"):
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            converter = get_converter()
            result = converter.convert(str(tmp_path))
            return result.document.export_to_markdown()
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        raise ValueError(
            f"Unsupported file format for conversion: {ext}. "
            f"Supported extensions: {', '.join(SUPPORTED_CONVERTER_EXTENSIONS)}"
        )


def convert_to_parsed(file_path: Path | str) -> Parsed:
    """Convert a file into a Parsed dataclass instance suitable for the ingest pipeline."""
    record = convert_file(file_path)
    return Parsed(text=record["text"], filename=record["filename"])


def convert_directory(
    input_dir: Path | str,
    output_file: Path | str | None = None,
    extensions: Sequence[str] = SUPPORTED_CONVERTER_EXTENSIONS,
) -> list[dict[str, str]]:
    """Convert all matching files in input_dir into a list of record dicts,

    and optionally save them to output_file in JSON format.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    records: list[dict[str, str]] = []

    for file_path in sorted(input_path.glob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in allowed:
            continue
        try:
            record = convert_file(file_path)
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF, DOCX, and TXT files to JSON using Docling."
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
    parser.add_argument("files", nargs="*", help="Optional specific file(s) to convert")

    args = parser.parse_args()

    if args.files:
        records = []
        for file_arg in args.files:
            try:
                record = convert_file(file_arg)
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
        records = convert_directory(args.input_dir, args.output)
        print(f"\nDone. {len(records)} file(s) converted -> {args.output}")


if __name__ == "__main__":
    main()
