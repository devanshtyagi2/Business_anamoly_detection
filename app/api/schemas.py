from pydantic import BaseModel

class FraudRequest(BaseModel):
    TransactionAmt: float
    card_tx_count: float
    card_amt_mean: float
    amt_over_card_mean: float
    anomaly_score: float
