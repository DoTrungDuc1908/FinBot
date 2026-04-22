"""
main.py — FinBot entrypoint
Starts the FastAPI server via uvicorn.
"""
import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        workers=1 if settings.api_debug else 2,
        log_level=settings.log_level.lower(),
    )
