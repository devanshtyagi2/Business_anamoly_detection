import pandas as pd
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier

PROCESSED_PATH = "data/processed/tx_feat.csv"
TARGET = "isFraud"

FEATURES = [
    "TransactionAmt",
    "card_tx_count",
    "card_amt_mean",
    "amt_over_card_mean",
    "anomaly_score",
]

def train_model():

    df = pd.read_csv(PROCESSED_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    mlflow.set_experiment("Business_Anomaly_Detection")

    with mlflow.start_run(run_name="LightGBM_Fraud_Model"):

        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X, y)

        mlflow.log_metric("train_accuracy", model.score(X, y))

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="FraudDetectionModel"
        )

        print("✅ Model trained & registered successfully")

if __name__ == "__main__":
    train_model()
