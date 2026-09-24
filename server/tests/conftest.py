import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

# Ensure local repository's forge package takes precedence
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("forge-data")
    os.environ["FORGE_DATA_DIR"] = str(directory)
    os.environ["FORGE_DATABASE_URL"] = f"sqlite:///{directory / 'test.db'}"

    from forge.config import get_settings

    get_settings.cache_clear()
    return directory


@pytest.fixture(scope="session")
def engine(data_dir: Path):
    from forge.db import get_engine, reset_engine
    from forge.models import Base

    reset_engine()
    eng = get_engine()
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def db(engine) -> Iterator[Session]:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM checkpoints"))
        conn.execute(text("DELETE FROM runs"))
        conn.execute(text("DELETE FROM models"))
        conn.execute(text("DELETE FROM jobs"))
        conn.execute(text("DELETE FROM datasets"))
        conn.execute(text("DELETE FROM documents"))
        conn.execute(text("DELETE FROM buckets"))
    maker = sessionmaker(bind=engine, expire_on_commit=False)
    session = maker()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(engine, db) -> TestClient:
    from forge.main import create_app

    return TestClient(create_app())


@pytest.fixture()
def make_document(db):
    # fake row, no files on disk

    def _make(bucket_id: uuid.UUID, chars: int = 100, status: str = "parsed"):
        from forge.models import Document

        doc = Document(
            bucket_id=bucket_id,
            filename=f"{uuid.uuid4().hex[:8]}.txt",
            source_format="txt",
            raw_uri=f"raw/{uuid.uuid4()}.txt",
            status=status,
            content_hash=uuid.uuid4().hex,
            char_count=chars,
            word_count=max(1, chars // 5),
            quality={},
        )
        db.add(doc)
        db.commit()
        return doc

    return _make


@pytest.fixture()
def make_dataset(db):
    def _make(name: str | None = None, bucket_id: uuid.UUID | None = None, status: str = "draft"):
        from forge.models import Dataset

        ds = Dataset(
            name=name or f"dataset-{uuid.uuid4().hex[:8]}",
            description="Test dataset",
            bucket_id=bucket_id,
            status=status,
            doc_count=10,
            char_count=5000,
            token_count=1250,
            config={"val_split": 0.1},
        )
        db.add(ds)
        db.commit()
        return ds

    return _make


@pytest.fixture()
def make_job(db):
    def _make(job_type: str = "tokenization", status: str = "pending"):
        from forge.models import Job

        job = Job(
            job_type=job_type,
            status=status,
            payload={"task": "benchmark"},
            progress={"step": 0},
        )
        db.add(job)
        db.commit()
        return job

    return _make


@pytest.fixture()
def make_model(db):
    def _make(name: str | None = None, preset: str = "nano"):
        from forge.models import Model

        model = Model(
            name=name or f"model-{uuid.uuid4().hex[:8]}",
            description="Test tiny LM",
            preset=preset,
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            dropout=0.0,
            param_count=920000,
            config={},
        )
        db.add(model)
        db.commit()
        return model

    return _make


@pytest.fixture()
def make_run(db, make_model, make_dataset):
    def _make(model_id: uuid.UUID | None = None, dataset_id: uuid.UUID | None = None, status: str = "pending"):
        from forge.models import Run

        if model_id is None:
            model = make_model()
            model_id = model.id

        run = Run(
            name=f"run-{uuid.uuid4().hex[:8]}",
            model_id=model_id,
            dataset_id=dataset_id,
            status=status,
            device="cpu",
            batch_size=2,
            learning_rate=0.001,
            max_steps=500,
            current_step=0,
            loss=2.5,
            metrics={"step_time_ms": 12.5},
            hyperparams={"precision": "fp32"},
        )
        db.add(run)
        db.commit()
        return run

    return _make


@pytest.fixture()
def make_checkpoint(db, make_run):
    def _make(run_id: uuid.UUID | None = None, step: int = 100, is_best: bool = False):
        from forge.models import Checkpoint

        if run_id is None:
            run = make_run()
            run_id = run.id

        ckpt = Checkpoint(
            run_id=run_id,
            step=step,
            epoch=1,
            loss=2.1,
            val_loss=2.0,
            uri=f"checkpoints/{run_id}/step_{step}.pt",
            size_bytes=3680000,
            is_best=is_best,
            metrics={"val_loss": 2.0},
        )
        db.add(ckpt)
        db.commit()
        return ckpt

    return _make


def _make_pdf(lines: list[str]) -> bytes:
    # minimal pdf with a text layer
    content = "BT /F1 12 Tf 72 720 Td 14 TL\n"
    for line in lines:
        content += f"({line}) Tj T*\n"
    content += "ET"
    stream = content.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


ARTICLE_PDF = _make_pdf(
    [
        "The quick brown fox jumps over the lazy dog. This page has a real text",
        "layer, so pypdf can pull the words straight back out of it without any",
        "optical character recognition at all. A scanned page has no text layer",
        "and comes out nearly empty, which is what the other fixture stands for.",
        "A second sentence follows here so the document clears the length floor",
        "that the scanned-PDF check uses to spot an image-only page.",
    ]
    * 3
)
# nearly empty, like a scan
SCANNED_PDF = _make_pdf(["12"])

# trailing spaces below are deliberate
NOTES_MD = (
    b"# Cleaning notes\n"
    b"\n"
    b"Markdown is kept **as-is**, the syntax is fine training text and stripping it\n"
    b"would be a decision we cannot undo later.\n"
    b"\n"
    b"## What we do change\n"
    b"\n"
    b"- collapse runs of blank lines\n"
    b"- strip trailing whitespace   \n"
    b"- drop repeated boilerplate lines\n"
    b"\n"
    b"That is all.\n"
)
