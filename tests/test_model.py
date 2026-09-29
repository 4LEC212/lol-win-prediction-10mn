import pytest

from lol_win.data import TARGET
from lol_win.model import evaluate_model, feature_importance, predict_blue_win_proba


def test_scores_match_notebook(trained_model):
    scores = evaluate_model(*trained_model)
    # 72% accuracy and 0.79 ROC-AUC in the notebook
    assert scores["accuracy"] == pytest.approx(0.72, abs=0.01)
    assert scores["roc_auc"] == pytest.approx(0.79, abs=0.01)


def test_evaluate_model_outputs(trained_model):
    _, _, y_test = trained_model
    scores = evaluate_model(*trained_model)
    assert scores["confusion_matrix"].sum() == len(y_test) == 1879
    assert scores["roc_curve"].iloc[0].tolist() == [0, 0]
    assert scores["roc_curve"].iloc[-1].tolist() == [1, 1]


def test_feature_importance(trained_model):
    model, _, _ = trained_model
    importance = feature_importance(model)
    assert len(importance) == 19
    assert importance.index[0] == "blueGoldDiff"
    assert importance["blueGoldDiff"] > 0


def test_predict_blue_win_proba(trained_model, clean_games):
    model, _, _ = trained_model
    median_game = clean_games.drop(columns=TARGET).median().to_dict()
    ahead = predict_blue_win_proba(model, {**median_game, "blueGoldDiff": 5000})
    behind = predict_blue_win_proba(model, {**median_game, "blueGoldDiff": -5000})
    assert 0 <= behind < 0.5 < ahead <= 1
