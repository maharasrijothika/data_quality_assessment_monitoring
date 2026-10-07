from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine, ensure_sqlite_columns
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
from app.routers.stages import router as stages_router
from app.routers.semantic import router as semantic_router
from app.routers.kb import router as kb_router
from app.routers.relationships import router as relationships_router
from app.routers.rules import router as rules_router
from app.routers.remediation import router as remediation_router
from app.routers.monitoring import router as monitoring_router
from app.routers.models import router as models_router
from app.routers.datasets import router as datasets_router

Base.metadata.create_all(bind=engine)
ensure_sqlite_columns()

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
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(datasets_router)
app.include_router(context_router)
app.include_router(profiling_router)
app.include_router(stages_router)
app.include_router(semantic_router)
app.include_router(kb_router)
app.include_router(relationships_router)
app.include_router(rules_router)
app.include_router(remediation_router)
app.include_router(monitoring_router)
app.include_router(models_router)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DQ Assessment API",
    }
