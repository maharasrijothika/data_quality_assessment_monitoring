from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    FileMetadata,
    TableMetadata,
    SemanticConcept,
    SemanticConceptEmbedding,
)
from app.routers.context import router as context_router
from app.routers.upload import router as upload_router
from app.routers.profiling import router as profiling_router
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Intelligent Data Quality Assessment System",
    version="1.0.0",
    openapi_version="3.0.3",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
app.include_router(upload_router)
app.include_router(context_router)
app.include_router(profiling_router)
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DQ Assessment API",
    }