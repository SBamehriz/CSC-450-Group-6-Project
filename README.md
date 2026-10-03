# Forge: Tiny-LM Data Lab

### CSC 450 Group 6 Project

Forge reads uploaded files, cleans their text, and organizes documents into collections. The app uses React, TypeScript, FastAPI, and SQLite. A separate PyTorch script benchmarks small language models.

---

## Features

- **Document Uploads**: Read files, clean text, and catch duplicates within a collection.
- **Document Review**: Search, filter, preview, move, reject, and delete documents.
- **Dashboard**: View collection totals, quality flags, and recent activity in a maroon interface.
- **Optional OCR**: Read images and scanned PDFs with RapidOCR, Docling, or a configured Tesseract setup.
- **Model Benchmarks**: Test speed and memory with the nano, micro, and small presets.

## Supported Formats

| Format | Method |
| :--- | :--- |
| TXT, Markdown | Text decoding |
| HTML | Trafilatura |
| PDF | pypdf; optional OCR for scans |
| DOCX | Native OpenXML; optional Docling fallback |
| Images | Optional OCR |
| JSON, JSONL | Text records |

## Installation & Setup

Install Python 3.11+, Node.js, and uv. From the project root:

```bash
uv sync --project server
uv run --project server python scripts/start.py --rebuild
```

The launcher runs database migrations, builds the frontend, and opens http://127.0.0.1:8000. Use `--rebuild` after frontend changes or `--dev` for live frontend updates. API docs are at `/api/docs`.

Optional dependencies:

```bash
uv sync --project server --extra ocr
uv sync --project server --extra converter
uv sync --project server --extra train
```

Choose the extras you need. Combine flags to keep multiple extras installed. OCR models may download on first use.

## Converter Usage

From `server`:

```bash
uv run python -m forge.converter -i ./documents -o ./dataset.json --clean
```

See [server/forge/README.md](server/forge/README.md) for converter options and Python examples.

## Current Status

Dataset snapshot creation and ZIP downloads need repair after the latest merge. The Python lockfile also needs updating. Tokenization and model training from uploaded documents are still future work.

## Running Tests

Backend, from `server`:

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

Frontend, from `web`:

```bash
npm ci
npm run lint
npm run format:check
npm test
npm run build
```

## Team

- Salim Bamehriz: frontend and app integration.
- Sampath Peddagolla: converter and database work.
- Sackey Ishmael: Python backend and API work.

## License

Developed for CSC 450. All rights reserved by Group 6.
