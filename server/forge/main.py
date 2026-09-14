from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from forge import buckets, documents
from forge.db import get_session
from forge.errors import ApiError, register_error_handlers
from forge.schemas import Health

API_PREFIX = "/api"

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="Forge", version="0.1.0", docs_url=f"{API_PREFIX}/docs")
    register_error_handlers(app)

    api = APIRouter(prefix=API_PREFIX)

    @api.get("/health", response_model=Health, tags=["health"])
    def get_health(session: Session = Depends(get_session)) -> Health:
        try:
            session.execute(text("SELECT 1"))
            database = "ok"
        except Exception:
            database = "unreachable"
        return Health(status="ok", database=database)

    api.include_router(buckets.router)
    api.include_router(documents.router)
    app.include_router(api)

    _serve_interface(app)
    return app


def _serve_interface(app: FastAPI) -> None:
    if not WEB_DIST.is_dir():
        return  # dev mode, vite serves it

    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def interface(full_path: str) -> FileResponse:
        # keep the catch-all off api urls
        if f"/{full_path}".startswith(f"{API_PREFIX}/"):
            raise ApiError(404, "not_found", "No such endpoint.")

        candidate = WEB_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")


app = create_app()
