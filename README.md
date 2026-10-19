# Forge: Tiny-LM Data Lab

### CSC 450 Group 6 Project

Forge reads uploaded files, cleans their text, and organizes documents into collections. The app uses React, TypeScript, FastAPI, and SQLite. A separate PyTorch script benchmarks small language models.

---

## Features

- **Document Uploads**: Read files, clean text, and catch duplicates within a collection.
- **Document Review**: Search, filter, preview, move, reject, and delete documents.
- **Dashboard**: View collection totals, quality flags, and recent activity in a maroon interface.
- **Optional OCR**: Read images and scanned PDFs with RapidOCR, Docling, or a configured Tesseract setup.
- **Model Sizes**: See the nano, micro, small, large, and 135m sizes, the memory each needs to train, and which fit this computer's graphics card.

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
uv sync --project server --extra prepare
```

Choose the extras you need. Combine flags to keep multiple extras installed. OCR models may download on first use.

## Converter Usage

From `server`:

```bash
uv run python -m forge.converter -i ./documents -o ./dataset.json --clean
```

See [server/forge/README.md](server/forge/README.md) for converter options and Python examples.

## Current Status

Text snapshots now save cleaned documents into a ZIP with train.jsonl, validation.jsonl, and manifest.json. Splits use whole documents and a fixed seed. Downloads check the saved hashes, and snapshots keep working after source files are removed. Old dataset records without a ZIP need a new snapshot. The prepare extra learns a BPE vocabulary from the training split, writes checked token files, and lets you try the tokenizer. The model now comes in five sizes, from nano (about 3M parameters) to 135m (about 134M parameters, 17 blocks, 768 wide, a context of 1,024). The Models page lists every size with its parameter count and the graphics card memory training it needs, and checks this computer's own card: its name, memory, driver, and whether PyTorch can use it. Each size is marked as fitting that card or not, so the answer changes with the machine. Model training controls are still planned.

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

## Prepare data

Open Datasets, build a text snapshot, then choose Prepare data. The vocabulary learns from training documents only. The prepared ZIP holds train.bin, validation.bin, tokenizer.json, and manifest.json. Tokens use unsigned 16-bit integers in little-endian order. Token 0 marks the end of each document.

This step uses the saved snapshot, even if its source files have changed or been deleted. Preparing is limited to 10 MB of text and 16,384 word pieces for now. The actual vocabulary can be smaller than requested. Install the prepare extra to use it. Model training is a later step.

## Sprint 6 dataset update

The training and validation counts and seed now describe real saved files. The split test uploads ten text files and checks that eight training documents and two validation documents appear in the downloaded ZIP. Dataset creation uses the current request fields. Small sources keep at least one document in each split.
