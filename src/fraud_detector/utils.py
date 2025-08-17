import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


def load_data():
    _ = load_dotenv()

    data_path = os.getenv("DATA_PATH")
    data_path = Path(data_path) if data_path else None
    print(f"Data path: {data_path}")

    if not data_path or not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    df = pd.read_csv(data_path, index_col=0)
    return df


def data_preprocessing(df):
    """
    Apply preprocesing steps raw data.
    Args:
        df (pd.DataFrame): Raw data DataFrame.
    """
    df = df.dropna(subset=["label"])

    # fill in NaN values for income using 0.
    # hypthesis. Missing income means no income reported.
    df["reported_income"] = df["reported_income"].fillna(0)

    # encode labels:
    df["label"] = df["label"].map({"CLEAN": 0, "EDITED": 1})

    return df
