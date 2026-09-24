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
