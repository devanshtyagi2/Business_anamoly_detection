import pandas as pd
import mlflow.pyfunc

MODEL_URI = "models/fraud_model"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("✅ Model loaded")
except Exception as e:
    model = None
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

    # MLflow pyfunc returns predictions using predict()
    preds = model.predict(df)

    # handle output safely
    if isinstance(preds[0], (list, tuple)):
        fraud_prob = float(preds[0][1])
    else:
        fraud_prob = float(preds[0])

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
