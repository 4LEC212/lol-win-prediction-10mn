import pytest

from lol_win.data import clean_data, load_data
from lol_win.model import train_model


@pytest.fixture(scope="session")
def raw_games():
    return load_data()


@pytest.fixture(scope="session")
def clean_games(raw_games):
    return clean_data(raw_games)


@pytest.fixture(scope="session")
def trained_model(clean_games):
    return train_model(clean_games)
