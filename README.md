# Forge: Tiny-LM Data Lab

### CSC 450 Group 6 Project

Forge reads uploaded files, cleans their text, and organizes documents into collections. The app uses React, TypeScript, FastAPI, and SQLite. It also trains, fine-tunes and compares small language models with PyTorch.

---

## Features

- **Document Uploads**: Read files, clean text, and catch duplicates within a collection.
- **Document Review**: Search, filter, preview, move, reject, and delete documents.
- **Dashboard**: View collection totals, quality flags, and recent activity in a maroon interface.
- **Optional OCR**: Read images and scanned PDFs with RapidOCR, Docling, or a configured Tesseract setup.
- **Model size by choice**: Say how many parameters you want, or how many hours you have. Forge plans the shape, the training length and the settings, says why, and checks it fits this computer's graphics card.
- **Fine-tune and chat**: Continue a model on conversations (a `messages` column from Hugging Face, or an uploaded `.jsonl` file with a `messages` list on each line), chat with it, and see two models answer side by side.

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

On Windows the PyPI torch is CPU only. To train on an NVIDIA card, run this after the sync, then start the app with `--no-sync` so `uv run` does not put the CPU build back:

```bash
uv pip install --python server/.venv/Scripts/python.exe --reinstall-package torch "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
uv run --project server --no-sync python scripts/start.py --rebuild
```

On Linux the PyPI torch already uses the card, so the sync alone is enough.

## Current Status

Text snapshots save cleaned documents as train.jsonl, validation.jsonl, and manifest.json; the ZIP is made when you download one. Splits use whole documents and a fixed seed. Downloads check the saved hashes, and snapshots keep working after source files are removed. Old dataset records without a ZIP need a new snapshot. The prepare extra learns a BPE vocabulary from the training split, writes checked token files, and lets you try the tokenizer. A model can be any size: the Train page plans one from a parameter count or the hours you have, and whether it fits is decided by the graphics card's free memory, not by a fixed limit.

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

Starting a training run now does this by itself: it makes a snapshot, a vocabulary and a training set when they are missing. The vocabulary learns from training documents only. The training set holds train.bin, validation.bin, and manifest.json. Tokens use unsigned 16-bit integers in little-endian order. Token 0 marks the end of each document.

This step uses the saved snapshot, even if its source files have changed or been deleted. The actual vocabulary can be smaller than requested. Install the prepare extra to use it.

## Sprint 6 dataset update

The training and validation counts and seed now describe real saved files. The split test uploads ten text files and checks that eight training documents and two validation documents appear in the downloaded ZIP. Dataset creation uses the current request fields. Small sources keep at least one document in each split.

## Training, measuring and comparing

- **Train.** On Train, pick a collection and a size or a time; Forge plans the rest and every number can be changed. To fine-tune, start from a model you trained. The run page shows the loss and validation loss as it trains, the speed and the time left, and has Stop, Resume and the save points. Batches depend only on the seed and the step, so a resumed run ends with exactly the same weights as one that never stopped. Save points are written to a temporary file first and moved into place, so a crash never leaves half of one. The best and the latest are kept. A run that goes NaN, runs out of card memory, fills the disk, or is cut off by the app stopping ends with its own plain reason.
- **Download.** Each save point downloads as a zip with the weights, the model's shape and its vocabulary, after its saved checksum is checked. Export for Hugging Face writes a GPT-2 folder that transformers loads with the same logits.
- **Settings.** Shows each card's total and free memory and runs a short speed test (under two minutes). The Train page uses it to estimate hours and memory for a size and a training length. Hours come from a speed test of that size, or one of another size scaled by the work per word piece, and the page says which. With no speed test it says so instead of guessing.
- **Import.** A model from Hugging Face is checked before anything downloads: safetensors only, a config and tokenizer present, the commit pinned, the licence recorded, room on the disk, and whether it fits the card. Each weight file is checked against the hub's sha256.
- **Compare.** Two models side by side: their size, vocabulary, training and loss, benchmark scores with a 95% interval and the chance level, and the same prompts answered with the same settings and seed. Benchmarks need 100 million parameters (ARC-Challenge and MMLU about 1 billion); below that they are refused with the reason, because the score would be chance-level noise. Bits per byte comes from Measure bits per byte on a run page, or from the Evaluation page, and shows "not measured yet" until it is measured. The same prompts can be typed in or taken from a saved prompt set.

Not checked yet: training, speed tests, import and benchmarks on a real graphics card, and import and benchmarks against the real Hugging Face hub. The tests use a stand-in for the card's driver tool and for the hub.

## Training from a terminal

For a graphics card with no browser (a lab machine reached over ssh, or a job queue), the `forge` command trains without the app open. It uses the same database and data folder, so a run started here shows up in the app like any other, with its charts and save points.

```bash
uv run --project server --no-sync forge info                  # card, free memory, PyTorch
uv run --project server --no-sync forge list                  # models, training sets, runs
uv run --project server --no-sync forge speed-test --sequences 8
uv run --project server --no-sync forge budget --params 135M --sequences 8 --hours 20
uv run --project server --no-sync forge train --name main --data "my text" --params 135M
uv run --project server --no-sync forge train --name main     # the same command resumes it
uv run --project server --no-sync forge sweep --model "tiny" --set "main set" --steps 300 --lr 3e-4,6e-4,1e-3 --batch 8,16
```

- `budget` turns the hours you have into word pieces: the measured speed times the hours, keeping a tenth for validation checks and save points. It prints the arithmetic and how many word pieces per parameter that is.
- `train` makes the run the first time and resumes it from its last save point after that, so one run can span several time-limited jobs. `--stop-after 3.5` saves and stops after 3.5 hours, before a job limit ends it; Ctrl+C or the queue's stop signal also saves first.
- `sweep` makes short runs that differ only in learning rate and batch size, trains them one after another and prints them ranked by validation loss.
- To move work to another computer, copy the whole `forge-data` folder there and back.

## Large amounts of text

Measured with 313 MB of text in seven 45 MB files (Linux, CPU, 7 GB of memory): upload took 56 s, the snapshot 6 s, a 16,384 word piece vocabulary 39 s, and the training set 83 s (69 million word pieces). The background process peaked at 2.5 GB. Text now reaches the vocabulary trainer and the tokenizer in pieces of about 100,000 characters; before that, the vocabulary step ran out of memory at 5.8 GB on the same files.

**Importing a Hugging Face dataset.** On a collection's page, Import from Hugging Face takes a dataset name, its configuration, split and column (text, or conversations such as `messages`), and how many rows to import (the whole split by default). The import reads the dataset's own data files from the hub one at a time, can be tried again where it stopped, drops empty and repeated rows, and groups rows so the collection gets at most about 2,000 files: a small import keeps one row per file, and 2 million short stories become about 2,000 files of about 1,000 stories each. It needs the prepare extra (pyarrow). A dataset with no Parquet copy on the hub falls back to the rows service, 100 rows a request, waiting and trying again when the hub says too many requests. Measured offline with 500,000 rows (390 MB of text): 51 s and 626 MB of memory.

A snapshot splits by file, so a few very large files make a poor validation split: seven files with 1% asked for gave one whole file, 14% of the text, to validation. Upload big text as many small files (about 1 MB each, up to 200 per upload) so the validation share comes out close to what you ask for. `scripts/split_text.py` does the cutting:

```bash
uv run --project server --no-sync python scripts/split_text.py TinyStoriesV2-GPT4-train.txt stories --size-mb 1
```
