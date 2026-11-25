from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOME = "conversion"


def load_criteo(path: str | Path, sample_n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def three_way_split(
    df: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random train/validation/test split, each stratified by treatment (the dataset has no dates).

    `val_size` is the share of all rows used for validation; the cutoff is chosen there.
    """
    rest, test = train_test_split(
        df, test_size=test_size, stratify=df[TREATMENT], random_state=RANDOM_STATE
    )
    train, val = train_test_split(
        rest,
        test_size=val_size / (1 - test_size),
        stratify=rest[TREATMENT],
        random_state=RANDOM_STATE,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
