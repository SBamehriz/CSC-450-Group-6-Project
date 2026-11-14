import os
from pathlib import Path
from typing import Any

# Suppress harmless Windows symlink warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from forge.ingest import (
    parse_docx_with_metadata,
    parse_image_with_metadata,
    parse_pdf_with_metadata,
)
from forge.ocr import get_ocr_engine_name, ocr_pdf

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")
SUPPORTED_CONVERTER_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    *SUPPORTED_IMAGE_EXTENSIONS,
)


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
