from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge.db import get_session
from forge.models import Document

router = APIRouter(tags=["overview"])

@router.get("/overview")
def get_overview(session: Session = Depends(get_session)):
    total_documents = session.scalar(
        select(func.count()).select_from(Document)
    ) or 0

    total_characters = session.scalar(
        select(func.coalesce(func.sum(Document.char_count), 0))
    ) or 0

    total_words = session.scalar(
        select(func.coalesce(func.sum(Document.word_count), 0))
    ) or 0

    return {
        "total_documents": total_documents,
        "total_characters": total_characters,
        "total_words": total_words,
    }