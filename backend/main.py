"""FastAPI application entry point for the SPINE backend API."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST

from .routes import work_router, status_router, audit_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SPINE Backend API",
        version="0.1.0",
        description="REST API for submitting and tracking SPINE work items",
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"detail": str(exc.errors())},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(work_router)
    app.include_router(status_router)
    app.include_router(audit_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
