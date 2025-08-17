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
    Preprocess the DataFrame by dropping NaN values in 'label' column
    and returning the cleaned DataFrame.
    """
    df = df.dropna(subset=["label"])  # Drop rows with NaN in 'label'
    return df
