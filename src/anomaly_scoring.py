import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from config import BASE_FEATURES, PROCESSED_PATH, RANDOM_STATE


def run_anomaly_detection():
    df = pd.read_csv(PROCESSED_PATH)

    X = df[BASE_FEATURES]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination=0.02,
        random_state=RANDOM_STATE
    )

    iso.fit(X_scaled)

    df["anomaly_score"] = -iso.decision_function(X_scaled)

    joblib.dump(iso, "models/anomaly.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    df.to_csv(PROCESSED_PATH, index=False)
    print("✅ Anomaly scoring completed")


if __name__ == "__main__":
    run_anomaly_detection()
