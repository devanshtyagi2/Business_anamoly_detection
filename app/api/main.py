from fastapi import FastAPI
from app.api.schemas import FraudRequest
from app.services.inference_service import predict

app = FastAPI(debug=True)  

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/predict")
def predict_fraud(payload: FraudRequest):
    print("✅ Received payload:", payload.dict())
    return predict(payload.dict())
