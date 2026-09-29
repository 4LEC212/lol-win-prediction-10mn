from streamlit.testing.v1 import AppTest


def run_app():
    return AppTest.from_file("../app.py", default_timeout=30).run()


def test_app_runs():
    at = run_app()
    assert not at.exception
    assert len(at.tabs) == 3


def test_gold_lead_increases_blue_win_probability():
    at = run_app()
    gold_slider = next(s for s in at.slider if s.label == "Blue gold difference")
    gold_slider.set_value(8000).run()
    blue_proba = next(m for m in at.metric if m.label == "Blue win probability")
    assert int(blue_proba.value.rstrip("%")) > 80
