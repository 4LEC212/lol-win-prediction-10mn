"""Final model of the notebook: logistic regression on the cleaned 10mn stats."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lol_win.data import split_features_target

RANDOM_STATE = 42


def train_model(df):
    """Train on 80% of the games (same split as the notebook), return the model and the test set."""
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # C comes from the RandomizedSearch in the notebook, scaling makes the coefficients comparable
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=7.4, solver="liblinear", max_iter=500, random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
    }


def feature_importance(model):
    """Coefficients on the standardized features: > 0 helps blue win, < 0 helps red."""
    coefs = model[-1].coef_[0]
    return pd.Series(coefs, index=model.feature_names_in_).sort_values(key=abs, ascending=False)


def predict_blue_win_proba(model, game):
    """Probability that blue wins, `game` being a dict {feature: value}."""
    X = pd.DataFrame([game])[model.feature_names_in_]
    return float(model.predict_proba(X)[0, 1])
