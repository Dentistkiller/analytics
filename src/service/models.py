from pydantic import BaseModel, Field

class ScoreRequest(BaseModel):
    tx_id: int = Field(..., gt=0, description="Transaction ID to score")

class ScoreResponse(BaseModel):
    tx_id: int
    score: float
    label_pred: bool
    threshold: float
    model: str | None = None
