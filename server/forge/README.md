# Forge Document Converter (`converter.py`)

A document conversion and ingestion utility for **Forge**, built to extract structured, high-fidelity Markdown text from PDF, Word, and text documents using [Docling](https://github.com/docling-project/docling).

This module can be used as both an **importable Python library** within the Forge backend ingestion pipeline and a **standalone CLI tool** for batch dataset preparation.

---

## Table of Contents

- [Features](#features)
- [Supported Formats](#supported-formats)
- [Installation & Setup](#installation--setup)
- [CLI Usage](#cli-usage)
- [Python API Usage](#python-api-usage)
- [Forge Ingestion Pipeline Integration](#forge-ingestion-pipeline-integration)
- [Output Schema](#output-schema)
- [Running Tests](#running-tests)
- [Git Commit & Push Guide](#git-commit--push-guide)

---

## Features

- **Multi-Format Extraction**: Parses `.pdf`, `.docx`, images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`), `.txt`, and `.md` files cleanly into Markdown text records.
- **Powered by Docling & RapidOCR**: Leverages Docling's advanced document layout analysis, table recognition, and OCR capabilities to preserve document structure and extract text from scanned PDFs and raster images.
- **Lazy Singleton Loading**: Heavy Docling models and dependencies are loaded lazily on demand (`get_converter()`), avoiding startup penalty for server routes or tasks that do not perform document conversions.
- **In-Memory & File-Based Ingestion**: Supports converting from filesystem paths (`convert_file`), directory trees (`convert_directory`), or raw byte streams (`convert_bytes` with automatic tempfile management).
- **Graceful Error Recovery**: Batch operations log and skip problematic files without aborting the entire pipeline.
- **Robust Encoding Support**: Handles standard UTF-8 as well as UTF-8 with BOM (`utf-8-sig`) transparently for plain text and Markdown files.
- **Direct Forge Pipeline Compatibility**: Generates `Parsed` dataclass objects compatible with Forge's document storage, cleaning, deduplication, and training buckets.

---

## Supported Formats

| Format | Extension | Engine / Method |
| :--- | :--- | :--- |
| **Portable Document Format** | `.pdf` | Native text extraction (`pypdf`) with Docling RapidOCR fallback for scanned pages |
| **Microsoft Word** | `.docx` | Native paragraph & table extraction (OpenXML) with Docling fallback -> Markdown |
| **Images** | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp` | Docling + RapidOCR (`DocumentConverter` -> Markdown) |
| **Markdown** | `.md` | Native `utf-8-sig` decoding |
| **Plain Text** | `.txt` | Native `utf-8-sig` decoding |

---

## Installation & Setup

Document conversion requires the optional `converter` extra (`docling>=2.126.0`).

### 1. Using `uv` (Recommended)

From the `server/` directory:

```bash
cd server
uv sync --extra converter
```

### 2. Using standard `pip`

Install the server package in editable mode with the converter extra:

```bash
cd server
pip install -e .[converter]
```

Or install `docling` directly:

```bash
pip install "docling>=2.126.0"
```

> **Note on First Run**: Docling and RapidOCR models download automatically on the first conversion (approx. 300–400 MB) and are cached locally for fast execution on subsequent runs.

---

## CLI Usage

You can invoke `converter.py` directly from the command line using Python module syntax or file execution.

### Basic Syntax

```bash
python -m forge.converter [OPTIONS] [FILES...]
```

### Command Options

- `-i, --input-dir <DIR>`: Input directory containing source documents (default: `input_files`).
- `-o, --output <FILE>`: Output path for the generated JSON file (default: `output.json`).
- `--ocr`: Force OCR processing on scanned PDFs and raster images (PNG, JPG, JPEG).
- `--metadata / --no-metadata`: Include structured extraction metadata (engine, page/table counts, dimensions) in records (default: enabled).
- `--clean`: Run extracted text through Forge's cleaning and quality validation pipeline.
- `--clean-mode {standard,none}`: Cleaning mode (default: `standard`).
- `--bucket <NAME_OR_UUID>`: Target Forge bucket to directly upload and ingest converted files.
- `--source-note <NOTE>`: Provenance note describing the source of uploaded documents.
- `[FILES...]`: Optional space-separated list of specific files to convert.

### Examples

#### 1. Batch Convert an Entire Directory with Cleaning

Convert all documents in a folder and run them through Forge's cleaning pipeline with quality metrics:

```bash
python -m forge.converter -i ./documents -o ./dataset.json --clean
```

#### 2. Convert Scanned PDFs and Images with OCR

Run OCR on scanned PDF contracts and JPG/PNG receipts:

```bash
python -m forge.converter contract_scan.pdf receipt.jpg --ocr -o ocr_results.json
```

#### 3. Convert and Upload Directly to a Bucket

Convert documents and ingest them directly into a Forge training bucket:

```bash
python -m forge.converter -i ./documents --bucket corpus --source-note "Q3 internal docs"
```

#### 4. Output Directly to Terminal (STDOUT)

Omit `-o` when passing files to print formatted JSON directly to standard output:

```bash
python -m forge.converter sample.txt
```

---

## Python API Usage

The module exposes programmatic functions for integration into custom scripts or server routes:

```python
from pathlib import Path
from forge.converter import (
    convert_file,
    convert_bytes,
    convert_to_parsed,
    convert_to_cleaned,
    convert_directory,
    convert_and_upload,
    convert_directory_and_upload,
)
from forge.ocr import ocr_image, ocr_pdf, ocr_document, is_ocr_available

# 1. OCR on PNG, JPG, or JPEG images
text_img = ocr_image("receipt.jpg")
print(text_img)

# 2. OCR on scanned PDFs
text_pdf = ocr_pdf("scanned_contract.pdf")
print(text_pdf)

# 3. Unified document OCR dispatcher
text_doc = ocr_document("scan.png")

# 4. Convert a single file from disk
record = convert_file("data/paper.pdf")
print(record["filename"])  # "paper.pdf"
print(record["text"])      # Extracted markdown string

# 2. Convert and run through Forge's cleaning pipeline
cleaned = convert_to_cleaned("data/paper.pdf")
print(cleaned.text)         # Cleaned markdown
print(cleaned.content_hash) # SHA-256 content hash
print(cleaned.quality)      # alpha_ratio, line_count, flags

# 3. Convert with inline cleaning metadata
record_cleaned = convert_file("data/paper.pdf", clean_text=True)
# record_cleaned contains: 'filename', 'text', 'content_hash', 'char_count', 'word_count', 'quality'

# 4. Convert in-memory bytes (e.g. from an HTTP file upload)
raw_bytes = Path("notes.docx").read_bytes()
markdown_text = convert_bytes("notes.docx", raw_bytes)

# 5. Convert to a Forge Parsed dataclass
parsed_doc = convert_to_parsed("data/article.md")

# 6. Convert and upload directly to a Forge bucket
doc = convert_and_upload("data/paper.pdf", bucket="corpus", source_note="uploaded via API")
print(f"Uploaded {doc.filename} -> {doc.status} (id: {doc.id})")

# 7. Batch convert a folder of documents
records = convert_directory(
    input_dir="./input_files",
    output_file="./output.json",
    clean_text=True,
)
```

---

## Forge Ingestion Pipeline Integration

Within Forge, `converter.py` powers rich document parsing in `forge.ingest`:

```python
from forge.ingest import parse_with_converter

# Extracts content using Docling and produces Parsed documents for Forge buckets
docs = parse_with_converter("uploaded_file.docx", raw_bytes)
for doc in docs:
    print(f"Ingested {doc.filename} ({len(doc.text)} characters)")
```

Documents converted through this pipeline undergo Forge's downstream text cleaning, quality scoring, alpha-ratio verification, deduplication, and tokenization for language model training.

---

## Output Schema

The output generated by `converter.py` is a JSON array of record objects with structured extraction metadata:

```json
[
  {
    "filename": "quarterly_review.pdf",
    "text": "## Q3 Financial Summary\n\nRevenue grew by **18%** year-over-year across enterprise software offerings...\n\n| Category | Growth |\n| :--- | :--- |\n| Subscriptions | +22% |\n| Services | +11% |",
    "metadata": {
      "source_format": "pdf",
      "engine": "pypdf",
      "ocr_applied": false,
      "page_count": 4,
      "char_count": 1840,
      "word_count": 290,
      "line_count": 42
    }
  },
  {
    "filename": "meeting_notes.docx",
    "text": "## Team Standup - Sept 14\n\n- Completed Docling integration for dataset pipeline\n- Verified unit tests for batch converter\n- Next steps: benchmark GPU training presets",
    "metadata": {
      "source_format": "docx",
      "engine": "openxml",
      "ocr_applied": false,
      "paragraph_count": 8,
      "heading_count": 2,
      "table_count": 1,
      "char_count": 920,
      "word_count": 145,
      "line_count": 18
    }
  },
  {
    "filename": "invoice_scan.png",
    "text": "ACME Supplies Inc.\nInvoice #88412\nTotal: $450.00",
    "metadata": {
      "source_format": "png",
      "engine": "RapidOCR",
      "ocr_applied": true,
      "width": 640,
      "height": 480,
      "image_format": "PNG",
      "image_mode": "RGB",
      "char_count": 68,
      "word_count": 8,
      "line_count": 3
    }
  }
]
```

---

## Running Tests

Unit tests for `converter.py` are located at `server/tests/test_converter.py`.

Run the test suite using `pytest`:

```bash
cd server
uv run pytest tests/test_converter.py
```

To run with verbose output:

```bash
cd server
uv run pytest tests/test_converter.py -v
```

---

## Git Commit & Push Guide

To commit this README and push it to GitHub:

```bash
# 1. Check current repository status
git status

# 2. Stage the new README file
git add server/forge/README.md

# 3. Commit the file with a clear message
git commit -m "docs: add comprehensive README for converter.py module"

# 4. Push changes to GitHub
git push origin main
```
