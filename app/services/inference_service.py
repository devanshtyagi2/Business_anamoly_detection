import pandas as pd
import mlflow.pyfunc
import os

# IMPORTANT — model must exist inside container
MODEL_URI = "models/fraud_model"

model = None

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("✅ Loaded model")
except Exception as e:
    print("❌ Model load failed:", e)

FEATURES = [
    "TransactionAmt",
    "card_tx_count",
    "card_amt_mean",
    "amt_over_card_mean",
    "anomaly_score",
]

def predict(payload: dict):

    global model

    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([[payload[f] for f in FEATURES]], columns=FEATURES)

    preds = model.predict(df)

    # probability handling
    fraud_prob = float(preds[0][1]) if len(preds.shape) > 1 else float(preds[0])
    fraud_prob = min(max(fraud_prob, 0.001), 0.999)

    if fraud_prob > 0.85:
        tier = "HIGH"
        action = "Block transaction"
    elif fraud_prob > 0.6:
        tier = "MEDIUM"
        action = "Send to manual review"
    else:
        tier = "LOW"
        action = "Allow transaction"

    return {
        "fraud_probability": round(fraud_prob, 4),
        "alert_tier": tier,
        "recommended_action": action,
    }
