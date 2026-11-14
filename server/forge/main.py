import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from forge import (
    buckets,
    chat_api,
    compare_api,
    datasets,
    documents,
    eval_api,
    jobs,
    jobs_api,
    measure_api,
    models_api,
    overview,
    plan_api,
    runs_api,
    training_api,
)
from forge.db import get_session
from forge.errors import ApiError, error_body, register_error_handlers
from forge.schemas import Health

API_PREFIX = "/api"

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "testserver"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Pending jobs survive closing the window or restarting the app.
    try:
        jobs.start_runner()
    except OSError:
        # A full process table must not hide the data and job controls too.
        logging.getLogger(__name__).exception("Could not start the background process")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Forge", version="0.1.0", docs_url=f"{API_PREFIX}/docs", lifespan=lifespan)
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
    api.include_router(datasets.router)
    api.include_router(overview.router)
    api.include_router(training_api.router)
    api.include_router(models_api.router)
    api.include_router(runs_api.router)
    api.include_router(jobs_api.router)
    api.include_router(eval_api.router)
    api.include_router(measure_api.router)
    api.include_router(compare_api.router)
    api.include_router(plan_api.router)
    api.include_router(chat_api.router)
    app.include_router(api)

    _guard(app)
    _serve_interface(app)
    return app


def _host_name(host: str) -> str:
    # "[::1]:8000" -> "[::1]", "localhost:8000" -> "localhost"
    if host.startswith("["):
        return host.split("]")[0] + "]"
    return host.split(":")[0]


def _guard(app: FastAPI) -> None:
    # on a shared server, start.py sets a token so only its owner gets in
    token = os.environ.get("FORGE_TOKEN")
    if not token:
        # no token: only pages on this computer, so another site can't reach the api
        @app.middleware("http")
        async def local_only(request: Request, call_next):
            if _host_name(request.headers.get("host", "").lower()) in LOCAL_HOSTS:
                return await call_next(request)
            message = "Open Forge at http://localhost on this computer."
            return JSONResponse(status_code=400, content=error_body("wrong_host", message))

        return

    def allowed(given: str | None) -> bool:
        return bool(given) and hmac.compare_digest(given.encode(), token.encode())

    @app.middleware("http")
    async def check(request: Request, call_next):
        given = request.query_params.get("token")
        if allowed(given):
            # open the link once, then the cookie carries it
            answer = RedirectResponse(request.url.remove_query_params("token"))
            answer.set_cookie("forge_token", token, httponly=True, samesite="strict")
            return answer
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        if allowed(request.cookies.get("forge_token")) or allowed(bearer):
            return await call_next(request)
        message = "Open Forge with the link its terminal printed."
        return JSONResponse(status_code=401, content=error_body("unauthorized", message))


def _serve_interface(app: FastAPI) -> None:
    if not WEB_DIST.is_dir():
        return  # dev mode, vite serves it

    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def interface(full_path: str) -> FileResponse:
        # keep the catch-all off api urls
        if f"/{full_path}".startswith(f"{API_PREFIX}/"):
            raise ApiError(404, "not_found", "No such endpoint.")

        root = WEB_DIST.resolve()
        candidate = (root / full_path).resolve()
        # only files inside the build, never ".." or an absolute path
        if full_path and candidate.is_relative_to(root) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")


app = create_app()
