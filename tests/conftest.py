import pytest

from lol_win.data import clean_data, load_data


@pytest.fixture(scope="session")
def raw_games():
    return load_data()


@pytest.fixture(scope="session")
def clean_games(raw_games):
    return clean_data(raw_games)
