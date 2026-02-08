import pandas as pd
import mlflow.pyfunc

# Load model from MLflow registry
MODEL_URI = "mlruns/538616375265912523/8c496f674292452587f4b4df835a8b09/artifacts/model"


model = mlflow.pyfunc.load_model(MODEL_URI)
print("✅ Loaded model")

# IMPORTANT: must match training FEATURES EXACTLY
FEATURES = [
    "TransactionAmt",
    "card_tx_count",
    "card_amt_mean",
    "amt_over_card_mean",
    "anomaly_score",
]


def predict(payload: dict):

    # Build dataframe in correct order
    df = pd.DataFrame([[payload[f] for f in FEATURES]], columns=FEATURES)

    # Get model output
    preds = model.predict(df)

    # ---- SAFE PROBABILITY EXTRACTION ----
    # If model returns probability array
    fraud_prob = float(preds[0][1]) if len(preds.shape) > 1 else float(preds[0])

# Cap probability slightly (production style)
    fraud_prob = min(max(fraud_prob, 0.001), 0.999)


    # ---- ALERT RULES ----
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
