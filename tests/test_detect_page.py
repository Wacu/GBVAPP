"""End-to-end checks on the detect page, driven through Streamlit's own harness.

These run the page the way the browser does -- module top to bottom, widgets
set, form submitted -- so they catch the wiring faults a unit test cannot: a
model missing from the dropdown, a branch that never fires, a crash when no API
key is set.

The API is not called. `apps.claude_classify.classify_batch` is replaced, so
these stay free and offline; what is under test is the page's wiring, not the
model.
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

# AppTest resolves a relative path against the *calling* file, so give it an
# absolute one from the repository root.
PAGE = str(Path(__file__).resolve().parent.parent / 'pages' / '03.GBV form detect.py')

CLAUDE_OPTION = 'Claude (Haiku 5.5)'


def fake_result(texts):
    """A classify_batch stand-in with the real column shape."""
    frame = pd.DataFrame({
        'text': list(texts),
        'is_violence': [True] * len(texts),
        'form': ['Physical Violence'] * len(texts),
        'text_type': ['disclosure'] * len(texts),
        'sentiment': ['negative'] * len(texts),
        'evidence': ['beat me'] * len(texts),
        'confidence': ['high'] * len(texts),
        'error': [None] * len(texts),
    })
    totals = {'input': 60, 'output': 120, 'cache_read': 1250, 'cache_write': 0,
              'calls': len(texts), 'errors': 0}
    return frame, totals


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    """Drop the cached API client between tests.

    `claude_client` is decorated with `st.cache_resource`, which outlives a
    single AppTest run. Without this, a client built by one test is handed to
    the next -- which silently defeated the no-key test.
    """
    import streamlit as st
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def page(monkeypatch):
    """The page with the API stubbed out and a key present."""
    import apps.claude_classify as cc

    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key-not-real')
    monkeypatch.setattr(cc, 'build_client', lambda api_key=None: object())
    monkeypatch.setattr(
        cc, 'classify_batch',
        lambda texts, client=None, model=cc.DEFAULT_MODEL, max_workers=1,
        progress=None: fake_result(texts))
    return AppTest.from_file(PAGE, default_timeout=60)


def test_page_loads_without_error(page):
    page.run()
    assert not page.exception


def test_claude_is_offered_alongside_the_trained_models(page):
    # The trained models must stay selectable: they are the whole app for
    # anyone with no API key, and they keep the comparison reproducible.
    page.run()
    options = page.selectbox[0].options
    assert 'Logistic Regression' in options
    assert 'SVM' in options
    assert CLAUDE_OPTION in options


@pytest.mark.slow
def test_trained_model_still_predicts_from_text_input(page):
    # The first trained-model prediction of a process loads the 66 MB GloVe
    # vectors, which takes well over a minute from cold. That is the app's
    # real behaviour, so the test waits rather than stubbing it away.
    page.default_timeout = 300
    page.run()
    page.text_area[0].set_value('he beat me last night')
    page.selectbox[0].set_value('SVM')
    # The widgets sit inside st.form, so nothing takes effect until the form
    # is submitted -- the same as clicking Predict in the browser.
    page.get('form_submit_button')[0].click()
    page.run()

    assert not page.exception
    assert any('Violence' in str(el.value) for el in page.success)


def test_choosing_claude_renders_the_structured_record(page):
    page.run()
    page.text_area[0].set_value('he beat me last night')
    page.selectbox[0].set_value(CLAUDE_OPTION)
    page.get('form_submit_button')[0].click()
    page.run()

    assert not page.exception
    assert len(page.dataframe) >= 1
    shown = page.dataframe[0].value
    # The fields the trained models cannot express are the point of this path.
    for column in ('is violence?', 'form', 'text type', 'evidence'):
        assert column in shown.columns


def test_claude_run_reports_its_cost(page):
    page.run()
    page.text_area[0].set_value('he beat me last night')
    page.selectbox[0].set_value(CLAUDE_OPTION)
    page.get('form_submit_button')[0].click()
    page.run()

    labels = [m.label for m in page.metric]
    assert 'Run cost' in labels


def test_missing_api_key_shows_a_reason_and_does_not_crash(monkeypatch):
    """The page must stay usable with no key, not traceback."""
    import apps.claude_classify as cc

    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setattr(cc, 'build_client',
                        lambda api_key=None: (_ for _ in ()).throw(
                            RuntimeError('ANTHROPIC_API_KEY is not set.')))

    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    at.selectbox[0].set_value(CLAUDE_OPTION)
    at.get('form_submit_button')[0].click()
    at.run()

    assert not at.exception
    assert any('ANTHROPIC_API_KEY' in str(e.value) for e in at.error)
