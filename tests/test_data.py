import numpy as np
import pandas as pd
import pytest

from lol_win.data import (
    COLUMNS_TO_DROP,
    TARGET,
    add_kills_assists,
    drop_redundant_columns,
    filter_games,
    load_data,
    remove_tower_outliers,
    remove_zscore_outliers,
    split_features_target,
)


def test_load_data(raw_games):
    assert raw_games.shape == (9879, 40)
    assert raw_games[TARGET].isin([0, 1]).all()


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")


def test_load_data_without_target(tmp_path):
    path = tmp_path / "games.csv"
    pd.DataFrame({"blueKills": [5, 3]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=TARGET):
        load_data(path)


def test_load_data_with_missing_values(tmp_path):
    path = tmp_path / "games.csv"
    pd.DataFrame({TARGET: [1, 0], "blueKills": [5, None]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing values"):
        load_data(path)


def test_drop_redundant_columns(raw_games):
    df = drop_redundant_columns(raw_games)
    assert df.shape == (9879, 22)
    assert not set(COLUMNS_TO_DROP) & set(df.columns)
    assert raw_games.shape == (9879, 40)  # input not modified


def test_remove_tower_outliers():
    df = pd.DataFrame({"blueTowersDestroyed": [0, 1, 2, 3, 4]})
    assert remove_tower_outliers(df)["blueTowersDestroyed"].tolist() == [0, 1, 2]


def test_remove_zscore_outliers():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "blueGoldDiff": rng.normal(0, 1000, 100),
        "blueDragons": rng.integers(0, 2, 100),
    })
    df.loc[0, "blueGoldDiff"] = 50_000  # way out of the distribution
    df.loc[1, "blueDragons"] = 10  # discrete column, not checked

    result = remove_zscore_outliers(df)
    assert 0 not in result.index
    assert 1 in result.index
    assert len(result) == 99


def test_add_kills_assists():
    df = pd.DataFrame({"blueKills": [5], "blueAssists": [7], "blueDeaths": [3], "redAssists": [2]})
    result = add_kills_assists(df)
    assert list(result.columns) == ["blueKillsAssists", "redKillsAssists"]
    assert result.iloc[0].tolist() == [12, 5]


def test_clean_data_matches_notebook(clean_games):
    # the notebook ends up with 9394 games, 19 features + the target
    assert clean_games.shape == (9394, 20)
    assert clean_games["blueTowersDestroyed"].max() <= 2


def test_split_features_target(clean_games):
    X, y = split_features_target(clean_games)
    assert X.shape == (9394, 19)
    assert TARGET not in X.columns
    assert y.name == TARGET


@pytest.mark.parametrize(
    "filters, check",
    [
        ({"first_blood": "blue"}, lambda df: (df["blueFirstBlood"] == 1).all()),
        ({"first_blood": "red"}, lambda df: (df["blueFirstBlood"] == 0).all()),
        ({"dragon": "blue"}, lambda df: ((df["blueDragons"] == 1) & (df["redDragons"] == 0)).all()),
        ({"dragon": "none"}, lambda df: ((df["blueDragons"] == 0) & (df["redDragons"] == 0)).all()),
        ({"herald": "red"}, lambda df: ((df["blueHeralds"] == 0) & (df["redHeralds"] == 1)).all()),
        ({"gold_diff": (-1000, 1000)}, lambda df: df["blueGoldDiff"].between(-1000, 1000).all()),
    ],
)
def test_filter_games(raw_games, filters, check):
    result = filter_games(raw_games, **filters)
    assert 0 < len(result) < len(raw_games)
    assert check(result)


def test_filter_games_combined(raw_games):
    result = filter_games(raw_games, first_blood="blue", dragon="blue", gold_diff=(0, 5000))
    assert len(result) > 0
    assert (result["blueFirstBlood"] == 1).all()
    assert (result["blueDragons"] == 1).all()
    assert result["blueGoldDiff"].between(0, 5000).all()


def test_filter_games_without_filters(raw_games):
    assert filter_games(raw_games).equals(raw_games)


def test_filter_games_no_match(raw_games):
    assert filter_games(raw_games, gold_diff=(50_000, 60_000)).empty
