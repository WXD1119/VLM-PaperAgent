try:
    from fastapi import FastAPI
except ImportError:  # optional dependency in the skeleton
    FastAPI = None  # type: ignore[assignment,misc]


def create_app():
    if FastAPI is None:
        raise RuntimeError("Install the api extra: pip install -e .[api]")
    app = FastAPI(title="VLM-PaperAgent API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app() if FastAPI is not None else None

