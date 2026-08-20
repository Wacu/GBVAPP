import streamlit as st
from pathlib import Path

from PIL import Image

CURRENT_DIR = Path(__file__).parent

# set_page_config must be the first Streamlit call in the script.
# The banner in assets/ is 300x168, too wide to read as a favicon, hence an
# emoji icon. layout="wide" matches the other pages.
st.set_page_config(
    page_title="GBV APP",
    page_icon="\U0001f4ca",
    layout="wide",
)

REPO_URL = "https://github.com/Wacu/GBVAPP"

# The four classes the trained models actually predict. The project description
# names five forms of GBV, but Harmful Traditional Practices is not among the
# model's classes -- listing it here would overstate what the app can do.
GBV_FORMS = [
    ("Economic Violence", "Control of money, property or the means to earn."),
    ("Emotional Violence", "Intimidation, humiliation, threats and coercion."),
    ("Physical Violence", "Bodily harm or the threat of it."),
    ("Sexual Violence", "Non-consensual sexual acts or coercion."),
]

PIPELINE = [
    (
        "1 - Select data",
        "pages/01.selectTweets.py",
        "Draw a sample of tweets from the stored corpus.",
    ),
    (
        "2 - Clean and explore",
        "pages/02.Cleaning_\U0001f610Sentiments_Topic.py",
        "Normalise the text, then score sentiment and surface common themes.",
    ),
    (
        "3 - Detect the form",
        "pages/03.GBV form detect.py",
        "Classify a tweet into one of the four forms above.",
    ),
]

# --- header -------------------------------------------------------------
intro, banner = st.columns([3, 2], gap="large")

with intro:
    st.title("GBV APP")
    st.subheader("Detecting forms of gender-based violence in short text")
    st.write(
        "A natural language processing pipeline that takes tweets about "
        "gender-based violence and classifies which **form** of violence each "
        "one describes -- alongside sentiment scoring and topic exploration of "
        "the same corpus."
    )

with banner:
    st.image(Image.open(CURRENT_DIR / "assets" / "GBV.png"), width=420)

st.divider()

# --- what it does -------------------------------------------------------
st.header("What this does")
st.write(
    "Reports of gender-based violence are often written as free text, which "
    "makes them hard to count and compare. This app turns that text into "
    "structured categories: each tweet is cleaned, converted into a numeric "
    "representation using pretrained word vectors, and then labelled with the "
    "form of violence it most closely describes. Sentiment scores and topic "
    "terms provide context for what a body of tweets is saying overall."
)

# --- forms --------------------------------------------------------------
st.header("Forms the model detects")
st.caption(
    "Four classes, as trained. A confidence score accompanies each prediction "
    "on the detection page."
)

for column, (name, description) in zip(
    st.columns(len(GBV_FORMS), gap="medium"), GBV_FORMS
):
    with column:
        st.markdown(f"**{name}**")
        st.caption(description)

st.divider()

# --- pipeline -----------------------------------------------------------
st.header("How it works")
st.caption(
    "Three steps, in order. Each writes its output back to the database for "
    "the next one to pick up."
)

for column, (label, page, description) in zip(
    st.columns(len(PIPELINE), gap="medium"), PIPELINE
):
    with column:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(description)
            st.page_link(page, label="Open")

st.divider()

# --- techniques ---------------------------------------------------------
st.header("Under the hood")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Text representation**")
    st.markdown(
        "- `glove-wiki-gigaword-50` pretrained word vectors, 400k vocabulary\n"
        "- Bag-of-words feature extraction over the cleaned corpus\n"
        "- Stopword removal, lemmatisation and stemming via NLTK"
    )
with right:
    st.markdown("**Modelling**")
    st.markdown(
        "- Logistic regression, scored one-vs-rest\n"
        "- Linear support vector machine, scored one-vs-one\n"
        "- VADER and TextBlob for sentiment; n-gram frequency for topics"
    )

# --- caveat -------------------------------------------------------------
st.info(
    "**On interpreting the output.** The classifiers were trained in 2020 and "
    "their accuracy has not been re-validated against a labelled test set. "
    "Predictions illustrate the method rather than provide a reliable "
    "measurement, and should not be used to draw conclusions about real cases.",
    icon=":material/info:",
)

st.divider()
st.caption(f"Built by Wacu Mutahi &nbsp;·&nbsp; [source on GitHub]({REPO_URL})")
