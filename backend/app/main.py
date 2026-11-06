from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health
from app.core.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered research assistant using Retrieval-Augmented Generation",
    version="1.0.0",
    debug=settings.DEBUG,
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API Routers
app.include_router(health.router)
app.include_router(documents.router)


@app.get("/")
def root():
    return {
        "message": "Welcome to AI Research Assistant API",
        "docs": "/docs",
        "health": "/health",
    }