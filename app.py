import pandas as pd
import plotly.express as px
import streamlit as st

from lol_win.data import TARGET, clean_data, filter_games, load_data
from lol_win.model import evaluate_model, feature_importance, predict_blue_win_proba, train_model

BLUE, RED = "#1f77b4", "#d62728"

# Test accuracies reported in the notebook (same test set of 1879 games)
NOTEBOOK_RESULTS = [
    ("Random Forest", "Polynomial features, 4 best", "69%"),
    ("XGBoost", "Polynomial features, 4 best", "70%"),
    ("SVM", "Polynomial features, 4 best", "71%"),
    ("Logistic Regression", "Polynomial features, 4 best", "71%"),
    ("AdaBoost", "Polynomial features, 4 best", "72%"),
    ("SVM tuned (linear, C=0.01)", "Polynomial features, 95 best", "71.5%"),
    ("Logistic Regression tuned (C=7.4)", "Polynomial features, 67 best", "71.6%"),
    ("Logistic Regression (C=7.4)", "19 cleaned features", "72.2%"),
]

st.set_page_config(page_title="LoL win prediction", page_icon="⚔️", layout="wide")


@st.cache_data
def get_games():
    raw_games = load_data()
    return raw_games, clean_data(raw_games)


@st.cache_resource
def get_model():
    _, games = get_games()
    model, X_test, y_test = train_model(games)
    return model, evaluate_model(model, X_test, y_test)


def to_filter(choice):
    """Selectbox value -> filter_games argument ("Any" means no filter)."""
    return None if choice == "Any" else choice.lower()


raw_games, games = get_games()
model, scores = get_model()

st.title("League of Legends: who wins after 10 minutes?")
st.write(
    f"{len(raw_games):,} ranked games of high elo players (Diamond / Master) with the stats of both teams "
    "at 10 minutes. Explore the games, check how the model performs, or predict the winner of a game."
)

explore_tab, model_tab, predict_tab = st.tabs(["Explore the games", "Model", "Predict a game"])

with explore_tab:
    col1, col2, col3, col4 = st.columns(4)
    first_blood = col1.selectbox("First blood", ["Any", "Blue", "Red"])
    dragon = col2.selectbox("Dragon", ["Any", "None", "Blue", "Red"])
    herald = col3.selectbox("Rift Herald", ["Any", "None", "Blue", "Red"])
    gold_diff = col4.slider("Blue gold difference range", -12000, 12000, (-12000, 12000), step=500)

    selected = filter_games(
        raw_games,
        first_blood=to_filter(first_blood),
        dragon=to_filter(dragon),
        herald=to_filter(herald),
        gold_diff=gold_diff,
    )

    if selected.empty:
        st.info("No game matches these filters.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Games", f"{len(selected):,}")
        col2.metric("Blue win rate", f"{selected[TARGET].mean():.1%}")
        col3.metric("Average blue gold difference", f"{selected['blueGoldDiff'].mean():+,.0f}")

        col1, col2 = st.columns(2)
        fig = px.histogram(
            selected.assign(Result=selected[TARGET].map({1: "Blue win", 0: "Red win"})),
            x="blueGoldDiff",
            color="Result",
            barmode="overlay",
            color_discrete_map={"Blue win": BLUE, "Red win": RED},
            labels={"blueGoldDiff": "Blue gold difference"},
            title="Gold difference at 10mn",
        )
        col1.plotly_chart(fig)

        towers = pd.crosstab(
            selected["redTowersDestroyed"],
            selected["blueTowersDestroyed"],
            values=selected[TARGET],
            aggfunc="mean",
        )
        fig = px.imshow(
            towers,
            text_auto=".0%",
            color_continuous_scale="RdBu",
            zmin=0,
            zmax=1,
            labels={"x": "Towers destroyed by blue", "y": "Towers destroyed by red", "color": "Blue win rate"},
            title="Blue win rate by towers destroyed",
        )
        fig.update_layout(xaxis_dtick=1, yaxis_dtick=1)  # whole numbers of towers only
        col2.plotly_chart(fig)

        st.dataframe(selected, hide_index=True)
        st.download_button("Download these games (CSV)", selected.to_csv(index=False), "games.csv", "text/csv")

with model_tab:
    st.write(
        f"Logistic regression trained on 80% of the {len(games):,} games left after removing outliers "
        "(same preprocessing and split as in the notebook), and evaluated on the other 20%."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Test accuracy", f"{scores['accuracy']:.1%}")
    col2.metric("ROC-AUC", f"{scores['roc_auc']:.2f}")
    col3.metric("Test games", f"{scores['confusion_matrix'].sum():,}")

    col1, col2 = st.columns(2)
    outcomes = ["Red win", "Blue win"]
    fig = px.imshow(
        scores["confusion_matrix"],
        x=outcomes,
        y=outcomes,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Games"},
        title="Confusion matrix",
    )
    col1.plotly_chart(fig)

    fig = px.area(
        scores["roc_curve"],
        x="fpr",
        y="tpr",
        labels={"fpr": "False positive rate", "tpr": "True positive rate"},
        title=f"ROC curve (AUC = {scores['roc_auc']:.2f})",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line_dash="dash")
    col2.plotly_chart(fig)

    importance = feature_importance(model)
    fig = px.bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        labels={"x": "Coefficient", "y": ""},
        title="Feature importance",
        height=550,
    )
    fig.update_traces(marker_color=[BLUE if coef > 0 else RED for coef in importance])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig)
    st.caption(
        "Coefficients of the logistic regression on standardized features, so they can be compared with each "
        "other: blue bars push towards a blue win, red bars towards a red win."
    )

    st.subheader("Models tried in the notebook")
    comparison = pd.DataFrame(
        NOTEBOOK_RESULTS + [("Logistic Regression + scaling (this app)", "19 cleaned features", f"{scores['accuracy']:.1%}")],
        columns=["Model", "Features", "Test accuracy"],
    )
    st.dataframe(comparison, hide_index=True)
    st.caption("Accuracies reported in the notebook on the same test set. The last row is computed by the app.")

with predict_tab:
    st.write("Set the state of a game at 10 minutes. Everything you don't change is the median game of the dataset.")
    game = games.drop(columns=TARGET).median().to_dict()

    col1, col2, col3 = st.columns(3)
    first_blood = col1.radio("First blood", ["Blue", "Red"], horizontal=True)
    dragon = col2.radio("Dragon", ["None", "Blue", "Red"], horizontal=True)
    herald = col3.radio("Rift Herald", ["None", "Blue", "Red"], horizontal=True)
    game["blueFirstBlood"] = int(first_blood == "Blue")
    game["blueDragons"], game["redDragons"] = int(dragon == "Blue"), int(dragon == "Red")
    game["blueHeralds"], game["redHeralds"] = int(herald == "Blue"), int(herald == "Red")

    col1, col2 = st.columns(2)
    game["blueGoldDiff"] = col1.slider("Blue gold difference", -10000, 10000, 0, step=100)
    game["blueExperienceDiff"] = col2.slider("Blue experience difference", -8000, 8000, 0, step=100)

    col1, col2, col3, col4 = st.columns(4)
    game["blueTowersDestroyed"] = col1.number_input("Towers destroyed by blue", 0, 2, 0)
    game["redTowersDestroyed"] = col2.number_input("Towers destroyed by red", 0, 2, 0)
    game["blueKillsAssists"] = col3.number_input("Blue kills + assists", min_value=0, value=int(game["blueKillsAssists"]))
    game["redKillsAssists"] = col4.number_input("Red kills + assists", min_value=0, value=int(game["redKillsAssists"]))

    with st.expander("More stats (wards and minions)"):
        col1, col2 = st.columns(2)
        for team, col in [("blue", col1), ("red", col2)]:
            for stat, label in [
                ("WardsPlaced", "wards placed"),
                ("WardsDestroyed", "wards destroyed"),
                ("TotalMinionsKilled", "minions killed"),
                ("TotalJungleMinionsKilled", "jungle minions killed"),
            ]:
                feature = team + stat
                game[feature] = col.number_input(f"{team.capitalize()} {label}", min_value=0, value=int(game[feature]))

    proba = predict_blue_win_proba(model, game)
    col1, col2 = st.columns(2)
    col1.metric("Blue win probability", f"{proba:.0%}")
    col2.metric("Red win probability", f"{1 - proba:.0%}")
    st.progress(proba)
