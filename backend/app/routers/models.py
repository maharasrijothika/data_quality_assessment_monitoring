from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.offline_learning import (
    list_models,
    promote_model,
    train_model,
)

router = APIRouter(tags=["models"])


class PromoteRequest(BaseModel):
    model_version_id: int


@router.post("/models/train")
def train(db: Session = Depends(get_db)):
    """Attempt offline training from accumulated human feedback."""
    try:
        result = train_model(db)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    db.commit()
    return result


@router.get("/models")
def get_models(db: Session = Depends(get_db)):
    return {"models": list_models(db)}


@router.post("/models/promote")
def promote(request: PromoteRequest, db: Session = Depends(get_db)):
    """Human-controlled promotion of a candidate model."""
    try:
        result = promote_model(db, request.model_version_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db.commit()
    return result
