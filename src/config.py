# Paths
RAW_TX_PATH = "data/raw/train_transaction.csv"
RAW_ID_PATH = "data/raw/train_identity.csv"
PROCESSED_PATH = "data/processed/tx_feat.csv"


BASE_FEATURES = [
    "TransactionAmt",
    "card_tx_count",
    "amt_over_card_mean",
    "tx_per_card_per_day",
    "time_since_last_tx",
    "card2_missing",
    "card5_missing",
    "addr1_missing"
]

TARGET = "isFraud"

RANDOM_STATE = 42
