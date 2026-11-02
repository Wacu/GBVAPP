import streamlit as st
st.set_page_config(layout='wide', page_title='Detect the form - GBV APP', page_icon='📊')


import pandas as pd

from apps.functions import *
from apps.cleaning import *
from apps.eda import *
from apps.sentiments import *
from apps.models import *
from apps.gbv_models import load_models
from apps.predict import predict_batch
from apps.claude_classify import (DEFAULT_MODEL, build_client, classify_batch,
                                  cost_of)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


tablename3 = 'cleaned'

# Load models

# models.npz is version-independent; see apps/gbv_models.py and
# tools/export_model_params.py for why the joblib pickle was retired.
loaded_models = load_models()

CLAUDE_OPTION = 'Claude (Haiku 5.5)'
MODEL_OPTIONS = ['Choose model', 'Logistic Regression', 'SVM', CLAUDE_OPTION]

# The SVM is the fallback when no model is chosen, matching the previous
# behaviour of this page.
def selected_model(choice):
    if choice == MODEL_OPTIONS[1]:
        return list(loaded_models.values())[0]
    return list(loaded_models.values())[1]


@st.cache_resource(show_spinner=False)
def claude_client():
    """Build the API client once per session.

    Returns (client, error message). The page must stay usable with no key:
    the trained models are the whole app for anyone without one.
    """
    try:
        return build_client(), None
    except Exception as err:
        return None, str(err)


def show_claude_results(results, totals, model):
    """Render a Claude run: the records, then what it cost."""
    failed = results[results['error'].notna()]
    if len(failed):
        st.warning('{} of {} rows could not be classified.'.format(
            len(failed), len(results)))
        with st.expander('Show the failures'):
            st.dataframe(failed[['text', 'error']], width='stretch')

    ok = results[results['error'].isna()]
    if ok.empty:
        return

    st.dataframe(
        ok.drop(columns='error').rename(columns={
            'text': 'tweet',
            'is_violence': 'is violence?',
            'text_type': 'text type',
        }),
        width='stretch',
    )

    violence = ok[ok['is_violence'] == True]  # noqa: E712 -- pandas mask
    left, right = st.columns(2)
    with left:
        st.metric('Rated as violence', '{} of {}'.format(len(violence), len(ok)))
        if len(violence):
            st.write('Forms found')
            st.bar_chart(violence['form'].value_counts())
    with right:
        st.metric('Run cost', '${:.4f}'.format(cost_of(totals, model)))
        st.write('Text types')
        st.bar_chart(ok['text_type'].value_counts())

    st.download_button(
        'Download results as CSV',
        ok.to_csv(index=False).encode('utf-8'),
        file_name='gbv_claude_results.csv',
        mime='text/csv',
    )


st.title('3 - Detect the form')
st.caption('Classify a tweet into one of four forms of gender-based violence. '
           'The confidence shown is the probability the model assigned to the '
           'label it predicted.')
st.write('\n')

choice = st.sidebar.selectbox('Detect by ', ['Text Input', 'Use Cleaned Data'])

if choice == 'Text Input':
    with st.form('my_form'):
        txt = st.text_area(
            label='Input a text',
            value=' Delete me and Paste your comments here and I will predict for you the form of GBV',
        )
        model_choice = st.selectbox('Trained  Classify Models Availabe', MODEL_OPTIONS)
        submitted = st.form_submit_button('Predict')

    if model_choice == MODEL_OPTIONS[0]:
        st.info('Select a model of choice')

    elif model_choice == CLAUDE_OPTION:
        client, err = claude_client()
        if err:
            st.error(err)
        elif submitted:
            with st.spinner('Asking Claude...'):
                results, totals = classify_batch([txt], client=client)
            show_claude_results(results, totals, DEFAULT_MODEL)

    else:
        try:
            results = predict_batch([txt], selected_model(model_choice))
        except ValueError as err:
            st.warning(str(err))
        else:
            st.subheader('Results')
            st.write('Prediction Labels')
            st.success(results['form'].iloc[0])
            st.write('Prediction Probability')
            st.info(results['confidence'].iloc[0])

if choice == "Use Cleaned Data":
    with st.form('cleaned'):
        model_choice2 = st.selectbox('Trained  Classify Models Availabe', MODEL_OPTIONS)
        # Claude is one API call per row, so the whole table is not the right
        # default for an interactive click. Cost is negligible either way;
        # this is about how long the page sits spinning.
        row_limit = st.number_input(
            'Rows to classify (Claude only)', min_value=1, max_value=500,
            value=25,
            help='The trained models always score the whole table -- they run '
                 'locally and take no time.',
        )
        submitted_clean = st.form_submit_button('Predict GBV form')

    if submitted_clean:
        data = read_selected_data(tablename3)

        if data is None or data.empty:
            st.warning(f'No rows found in the "{tablename3}" table.')

        elif model_choice2 == MODEL_OPTIONS[0]:
            st.info('Select a model of choice')

        elif model_choice2 == CLAUDE_OPTION:
            client, err = claude_client()
            if err:
                st.error(err)
            else:
                # Claude reads the raw tweet. apps/cleaning.py strips casing
                # and punctuation and rewrites some words, which removes the
                # negation and code-switching this depends on.
                usable = data[data['tweet'].notna()]
                usable = usable[usable['tweet'].astype(str).str.strip() != '']
                batch = usable['tweet'].head(int(row_limit)).tolist()

                st.subheader('Results')
                st.caption('Reading the raw tweets, not the cleaned column.')
                progress = st.progress(0.0, text='Starting...')

                def tick(done, total):
                    progress.progress(done / total,
                                      text='Classified {}/{}'.format(done, total))

                results, totals = classify_batch(
                    batch, client=client, max_workers=8, progress=tick)
                progress.empty()

                show_claude_results(results, totals, DEFAULT_MODEL)

        else:
            # lemma_nostops is the fully processed column the models expect;
            # clean_tweet is the fallback for rows written before it existed.
            source_column = 'lemma_nostops' if 'lemma_nostops' in data.columns else 'clean_tweet'

            # Drop rows whose cleaning removed every token before predicting, so
            # the predictions line up with the tweets they are shown against.
            usable = data[data[source_column].notna()]
            usable = usable[usable[source_column].astype(str).str.strip() != '']
            skipped = len(data) - len(usable)

            try:
                results = predict_batch(
                    usable[source_column].tolist(),
                    selected_model(model_choice2),
                )
            except ValueError as err:
                st.warning(str(err))
            else:
                results.insert(0, 'tweet', usable['tweet'].tolist())
                results = results.drop(columns='text')

                st.subheader('Results')
                st.write(f'Predicted the GBV form for {len(results)} of {len(data)} rows.')
                if skipped:
                    st.caption(f'{skipped} row(s) skipped -- cleaning left them with no words to score.')

                st.dataframe(results, width='stretch')

                st.write('Distribution of predicted forms')
                st.bar_chart(results['form'].value_counts())

                st.download_button(
                    'Download predictions as CSV',
                    results.to_csv(index=False).encode('utf-8'),
                    file_name='gbv_form_predictions.csv',
                    mime='text/csv',
                )
