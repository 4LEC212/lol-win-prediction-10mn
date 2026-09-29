# League of Legends Win Prediction with 10mn in-game stats

[![CI](https://github.com/4LEC212/lol-win-prediction-10mn/actions/workflows/ci.yml/badge.svg)](https://github.com/4LEC212/lol-win-prediction-10mn/actions/workflows/ci.yml)

<p align="center">
  <img src="images/feature_importance.png" alt="Feature importance" width="720">
</p>

This repository contains my **first end-to-end machine learning project** on real game data.  
The goal is to predict whether the **blue side** will win a League of Legends match using the game state after ~10 minutes (objectives, kills, towers, etc.).
The dataset consists of game data from high-elo players, which makes the outcomes of the games less random.

The analysis is in the notebook, and the project is now also a **Streamlit app**, packaged with **Docker**, tested with **pytest** and checked by **GitHub Actions**.

---

## The Streamlit app

The app has 3 tabs:

- **Explore the games**: filter the 9,879 games by first blood, dragon, Rift Herald and gold difference, and see the blue win rate, the gold difference of wins vs losses, the win rate by towers destroyed, and the filtered games (downloadable as CSV).
- **Model**: test accuracy, ROC-AUC, confusion matrix, ROC curve and feature importance of the final model, plus the models compared in the notebook.
- **Predict a game**: set the state of a game at 10 minutes (first blood, dragon, herald, gold / XP difference, towers, kills...) and get the probability that blue wins.

The model is the final logistic regression of the notebook. It is retrained when the app starts (less than a second), so no model file is stored in the repo.

---

## Project structure

```
├── app.py                      # Streamlit app
├── lol_win/
│   ├── data.py                 # data loading, cleaning and filtering
│   └── model.py                # training, evaluation and prediction
├── tests/                      # pytest tests
├── data/high_diamond_ranked_10min.csv
├── LoL_win_predict.ipynb       # original analysis
├── requirements.txt            # app dependencies (pinned)
├── requirements-dev.txt        # + tests and notebook dependencies
├── Dockerfile
└── .github/workflows/ci.yml    # CI: tests + Docker build
```

---

## Run the app locally

Requires Python 3.13.

```bash
git clone https://github.com/4LEC212/lol-win-prediction-10mn.git
cd lol-win-prediction-10mn
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens on http://localhost:8501.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the data loading / cleaning / filtering functions, the model (its scores must match the notebook) and the app itself (with Streamlit's `AppTest`).

## Run with Docker

Build and run the image locally:

```bash
docker build -t lol-win-predictor .
docker run --rm -p 8501:8501 lol-win-predictor
```

Or use the image published by the CI:

```bash
docker run --rm -p 8501:8501 ghcr.io/4lec212/lol-win-prediction-10mn:latest
```

Then open http://localhost:8501. The published image is built for `linux/amd64` (it also runs on Apple Silicon through Docker Desktop).

## CI

On every push and pull request, GitHub Actions:

1. installs the pinned dependencies and runs `pytest`,
2. builds the Docker image and checks that the app starts in the container,
3. on `main` only, pushes the image to the GitHub Container Registry (tagged `latest` and with the commit SHA).

## Reproducibility

- Python version fixed (3.13) in the Dockerfile, the CI and this README
- all dependencies pinned to exact versions
- fixed `random_state=42` for the train/test split and the model, so the app gets the same results as the notebook (72.2% accuracy)
- the model is rebuilt from the CSV at startup instead of loading a pickled model
- Docker images tagged with the commit they were built from

---

## Dataset & Objective

- **Source:** [League of Legends Diamond Ranked Games (10 min)](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min) on Kaggle.
- **Unit of prediction:** one match.
- **Target:** `blueWins` (`1` if blue team wins, `0` otherwise).
- **Features:** 10-minute game stats such as:
  - number of dragons taken
  - towers destroyed
  - first blood
  - kills/assists
  - blue vs red differences

The intuition: after 10 minutes, the game state already contains a lot of signal about who will win.

---

## Approach

1. **Data loading & cleaning**: load the CSV, remove redundant features and outliers (games where a team gave up).
2. **Feature engineering**: merge kills and assists, keep the gold and experience differences.
3. **Baselines**: start with a simple model to get a reference.
4. **Better models**: compare Random Forest, AdaBoost, Logistic Regression, XGBoost and SVM, then tune the best ones.
5. **Evaluation**: accuracy, ROC-AUC, and feature importance to understand what the model learns.

Because this is a **learning / showcase project**, the notebook also shows my thought process: I try things, compare models and leave a few alternative attempts. The idea is to show **how I would tackle an ML problem from scratch**, not just the final answer.

---

## Result of the best model

| Model | Accuracy | ROC-AUC |
|--------|-----------|----------|
| Logistic Regression | **0.72** | **0.79** |

---

## Run the notebook

```bash
pip install -r requirements-dev.txt
jupyter notebook LoL_win_predict.ipynb
```
