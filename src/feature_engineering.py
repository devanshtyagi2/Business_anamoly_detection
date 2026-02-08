import pandas as pd
from config import RAW_TX_PATH, RAW_ID_PATH, PROCESSED_PATH


def load_data():
    tx = pd.read_csv(RAW_TX_PATH)
    idn = pd.read_csv(RAW_ID_PATH)
    df = tx.merge(idn, on="TransactionID", how="left")
    return df


def add_missing_flags(df):
    for c in ["card2", "card5", "addr1"]:
        df[f"{c}_missing"] = df[c].isna().astype(int)
    return df


def card_aggregates(df):
    g = df.groupby("card1")["TransactionAmt"]
    df["card_tx_count"] = g.transform("count")
    df["card_amt_mean"] = g.transform("mean")
    df["card_amt_std"] = g.transform("std").fillna(0)
    df["amt_over_card_mean"] = df["TransactionAmt"] / (df["card_amt_mean"] + 1)
    return df


def time_velocity_features(df):
    df = df.sort_values("TransactionDT")

    df["time_since_last_tx"] = (
        df.groupby("card1")["TransactionDT"]
        .diff()
        .fillna(999999)
    )

    df["day"] = df["TransactionDT"] // 86400
    df["tx_per_card_per_day"] = (
        df.groupby(["card1", "day"])["TransactionAmt"]
        .transform("count")
    )
    return df


def build_features():
    df = load_data()
    df = add_missing_flags(df)
    df = card_aggregates(df)
    df = time_velocity_features(df)

    df.to_csv(PROCESSED_PATH, index=False)
    print("✅ Feature engineering completed")


if __name__ == "__main__":
    build_features()
