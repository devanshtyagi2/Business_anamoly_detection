from fastapi import APIRouter
from app.api.schemas import TransactionInput, PredictionOutput
from app.services.inference_service import run_inference

router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
def predict(tx: TransactionInput):
    return run_inference(tx.dict())
