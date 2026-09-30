from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_review_and_correction_render():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
