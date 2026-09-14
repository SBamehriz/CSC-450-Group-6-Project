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

- **Multi-Format Extraction**: Parses `.pdf`, `.docx`, `.txt`, and `.md` files cleanly into Markdown text records.
- **Powered by Docling**: Leverages Docling's advanced document layout analysis, table recognition, and OCR capabilities to preserve document structure.
- **Lazy Singleton Loading**: Heavy Docling models and dependencies are loaded lazily on demand (`get_converter()`), avoiding startup penalty for server routes or tasks that do not perform document conversions.
- **In-Memory & File-Based Ingestion**: Supports converting from filesystem paths (`convert_file`), directory trees (`convert_directory`), or raw byte streams (`convert_bytes` with automatic tempfile management).
- **Graceful Error Recovery**: Batch operations log and skip problematic files without aborting the entire pipeline.
- **Robust Encoding Support**: Handles standard UTF-8 as well as UTF-8 with BOM (`utf-8-sig`) transparently for plain text and Markdown files.
- **Direct Forge Pipeline Compatibility**: Generates `Parsed` dataclass objects compatible with Forge's document storage, cleaning, deduplication, and training buckets.

---

## Supported Formats

| Format | Extension | Engine / Method |
| :--- | :--- | :--- |
| **Portable Document Format** | `.pdf` | Docling (`DocumentConverter` -> Markdown) |
| **Microsoft Word** | `.docx` | Docling (`DocumentConverter` -> Markdown) |
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
- `[FILES...]`: Optional space-separated list of specific files to convert.

### Examples

#### 1. Batch Convert an Entire Directory

Convert all supported documents in a folder (`./documents`) and export results to `./dataset.json`:

```bash
python -m forge.converter -i ./documents -o ./dataset.json
```

#### 2. Convert Specific Files

Convert selected documents and save the resulting records to `output.json`:

```bash
python -m forge.converter report.pdf meeting_notes.docx -o output.json
```

#### 3. Output Directly to Terminal (STDOUT)

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
    convert_directory,
)

# 1. Convert a single file from disk
record = convert_file("data/paper.pdf")
print(record["filename"])  # "paper.pdf"
print(record["text"])      # Extracted markdown string

# 2. Convert in-memory bytes (e.g. from an HTTP file upload)
raw_bytes = Path("notes.docx").read_bytes()
markdown_text = convert_bytes("notes.docx", raw_bytes)

# 3. Convert to a Forge Parsed dataclass
parsed_doc = convert_to_parsed("data/article.md")
# parsed_doc.filename == "article.md"
# parsed_doc.text == "# Header..."

# 4. Batch convert a folder of documents
records = convert_directory(
    input_dir="./input_files",
    output_file="./output.json",
    extensions=(".pdf", ".docx", ".txt", ".md"),
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

The output generated by `converter.py` is a JSON array of record objects:

```json
[
  {
    "filename": "quarterly_review.pdf",
    "text": "## Q3 Financial Summary\n\nRevenue grew by **18%** year-over-year across enterprise software offerings...\n\n| Category | Growth |\n| :--- | :--- |\n| Subscriptions | +22% |\n| Services | +11% |"
  },
  {
    "filename": "meeting_notes.docx",
    "text": "## Team Standup - Sept 14\n\n- Completed Docling integration for dataset pipeline\n- Verified unit tests for batch converter\n- Next steps: benchmark GPU training presets"
  },
  {
    "filename": "guidelines.txt",
    "text": "Document curation guidelines for language model fine-tuning..."
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
