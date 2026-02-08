import mlflow.lightgbm

MODEL_NAME = "FraudDetectionModel"
MODEL_VERSION = 1

def load_model():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.lightgbm.load_model(model_uri)
    return model
