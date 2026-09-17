from fastapi import FastAPI

from app.database import Base, engine
from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    FileMetadata,
    TableMetadata,
)
from app.routers.upload import router as upload_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Intelligent Data Quality Assessment System",
    version="1.0.0",
    openapi_version="3.0.3",
)

app.include_router(upload_router)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DQ Assessment API",
    }