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
uv sync --project server --extra hub
uv sync --project server --extra bench
```

Choose the extras you need. Combine flags to keep multiple extras installed, because each `uv sync` removes the extras it was not given. For everything on this page: `uv sync --project server --extra prepare --extra train --extra hub --extra bench`. OCR models may download on first use.

- `train`: PyTorch, for training, speed tests and save points.
- `hub`: importing open models from Hugging Face, exporting to its format, and asking two models the same prompts.
- `bench`: benchmarks through EleutherAI's lm-evaluation-harness.

On Windows the PyPI torch is CPU only. To train on an NVIDIA card, run this after the sync:

```bash
uv pip install --project server torch --index-url https://download.pytorch.org/whl/cu124
```

## Converter Usage

From `server`:

```bash
uv run python -m forge.converter -i ./documents -o ./dataset.json --clean
```

See [server/forge/README.md](server/forge/README.md) for converter options and Python examples.

## Current Status

Text snapshots now save cleaned documents into a ZIP with train.jsonl, validation.jsonl, and manifest.json. Splits use whole documents and a fixed seed. Downloads check the saved hashes, and snapshots keep working after source files are removed. Old dataset records without a ZIP need a new snapshot. The prepare extra learns a BPE vocabulary from the training split, writes checked token files, and lets you try the tokenizer. The model now comes in five sizes, from nano (about 3M parameters) to 135m (about 134M parameters, 17 blocks, 768 wide, a context of 1,024). The Models page lists every size with its parameter count and the graphics card memory training it needs, and checks this computer's own card: its name, memory, driver, and whether PyTorch can use it. Each size is marked as fitting the memory that is free on that card right now, so the answer changes with the machine and with what else is running on it. Sizes go up to 48 blocks and 2,048 wide, about a billion parameters; whether one fits is decided by the card, not by a fixed limit.

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

This step uses the saved snapshot, even if its source files have changed or been deleted. Preparing is limited to 10 MB of text and 16,384 word pieces for now. The actual vocabulary can be smaller than requested. Install the prepare extra to use it.

## Sprint 6 dataset update

The training and validation counts and seed now describe real saved files. The split test uploads ten text files and checks that eight training documents and two validation documents appear in the downloaded ZIP. Dataset creation uses the current request fields. Small sources keep at least one document in each split.

## Training, measuring and comparing

- **Train.** On Training history, New run picks a model and a training set made with the same vocabulary, a length in word pieces or steps, and Advanced settings. The run page shows the loss and validation loss as it trains, the speed and the time left, and has Stop, Resume and the save points. Batches depend only on the seed and the step, so a resumed run ends with exactly the same weights as one that never stopped. Save points are written to a temporary file first and moved into place, so a crash never leaves half of one. The best and the latest are kept. A run that goes NaN, runs out of card memory, fills the disk, or is cut off by the app stopping ends with its own plain reason.
- **Download.** Each save point downloads as a zip with the weights, the model's shape and its vocabulary, after its saved checksum is checked. Export for Hugging Face writes a GPT-2 folder that transformers loads with the same logits.
- **This computer.** Shows each card's total and free memory, runs a short speed test (under two minutes), and estimates hours and memory for a size and a training length. Hours come from a speed test of that size, or one of another size scaled by the work per word piece, and the page says which. With no speed test it says so instead of guessing.
- **Import.** A model from Hugging Face is checked before anything downloads: safetensors only, a config and tokenizer present, the commit pinned, the licence recorded, room on the disk, and whether it fits the card. Each weight file is checked against the hub's sha256.
- **Compare.** Two models side by side: their size, vocabulary, training and loss, benchmark scores with a 95% interval and the chance level, and the same prompts answered with the same settings and seed. Benchmarks need 100 million parameters (ARC-Challenge and MMLU about 1 billion); below that they are refused with the reason, because the score would be chance-level noise. Bits per byte comes from Measure bits per byte on a run page, or from the Evaluation page, and shows "not measured yet" until it is measured. The same prompts can be typed in or taken from a saved prompt set.

Not checked yet: training, speed tests, import and benchmarks on a real graphics card, and import and benchmarks against the real Hugging Face hub. The tests use a stand-in for the card's driver tool and for the hub.
