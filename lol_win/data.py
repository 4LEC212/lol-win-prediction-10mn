"""Loading, cleaning and filtering of the 10mn game stats (same steps as in the notebook)."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "high_diamond_ranked_10min.csv"

TARGET = "blueWins"

# Columns already represented by other features (see the EDA in the notebook)
COLUMNS_TO_DROP = [
    "gameId", "redFirstBlood", "blueEliteMonsters", "redEliteMonsters", "redKills",
    "blueTotalExperience", "blueTotalGold", "redTotalExperience", "redTotalGold",
    "redDeaths", "blueAvgLevel", "redAvgLevel", "blueGoldPerMin", "redGoldPerMin",
    "blueCSPerMin", "redCSPerMin", "redGoldDiff", "redExperienceDiff",
]

# Discrete columns are left out of the z-score outlier removal
DISCRETE_COLUMNS = [
    "blueFirstBlood", "blueWins", "blueDragons", "blueHeralds",
    "redHeralds", "redDragons", "blueTowersDestroyed", "redTowersDestroyed",
]


def load_data(path=DATA_PATH):
    """Load the raw games (one row per game) and check the file is usable."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' not found in {path}")
    if df.isna().any().any():
        raise ValueError(f"Missing values found in {path}")
    return df


def drop_redundant_columns(df):
    return df.drop(columns=COLUMNS_TO_DROP)


def remove_tower_outliers(df, max_towers=2):
    """3 or 4 towers destroyed by blue at 10mn means the other team gave up."""
    return df[df["blueTowersDestroyed"] <= max_towers]


def remove_zscore_outliers(df, threshold=4):
    """Remove games where a continuous stat is more than `threshold` std away from the mean."""
    continuous_cols = [col for col in df.columns if col not in DISCRETE_COLUMNS]
    z_scores = np.abs(stats.zscore(df[continuous_cols]))
    return df[(z_scores < threshold).all(axis=1)]


def add_kills_assists(df):
    """Merge kills and assists (strongly correlated) into one feature per team."""
    df = df.copy()
    df["blueKillsAssists"] = df["blueKills"] + df["blueAssists"]
    df["redKillsAssists"] = df["blueDeaths"] + df["redAssists"]  # blueDeaths == redKills
    return df.drop(columns=["blueKills", "blueAssists", "blueDeaths", "redAssists"])


def clean_data(df):
    """Full preprocessing of the final model in the notebook (9879 -> 9394 games)."""
    df = drop_redundant_columns(df)
    df = remove_tower_outliers(df)
    df = remove_zscore_outliers(df)
    return add_kills_assists(df)


def split_features_target(df):
    return df.drop(columns=TARGET), df[TARGET]


def filter_games(df, first_blood=None, dragon=None, herald=None, gold_diff=None):
    """Keep the games matching an early game state (None = no filter).

    first_blood: "blue" or "red"
    dragon, herald: "blue", "red" or "none"
    gold_diff: (min, max) range of blueGoldDiff
    """
    mask = pd.Series(True, index=df.index)
    if first_blood is not None:
        mask &= df["blueFirstBlood"] == int(first_blood == "blue")
    for objective, team in [("Dragons", dragon), ("Heralds", herald)]:
        if team is not None:
            mask &= df[f"blue{objective}"] == int(team == "blue")
            mask &= df[f"red{objective}"] == int(team == "red")
    if gold_diff is not None:
        mask &= df["blueGoldDiff"].between(*gold_diff)
    return df[mask]
