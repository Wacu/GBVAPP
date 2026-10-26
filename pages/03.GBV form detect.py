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

MODEL_OPTIONS = ['Choose model', 'Logistic Regression', 'SVM']

# The SVM is the fallback when no model is chosen, matching the previous
# behaviour of this page.
def selected_model(choice):
    if choice == MODEL_OPTIONS[1]:
        return list(loaded_models.values())[0]
    return list(loaded_models.values())[1]


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
        submitted_clean = st.form_submit_button('Predict GBV form')

    if submitted_clean:
        data = read_selected_data(tablename3)

        if data is None or data.empty:
            st.warning(f'No rows found in the "{tablename3}" table.')
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

                st.dataframe(results, use_container_width=True)

                st.write('Distribution of predicted forms')
                st.bar_chart(results['form'].value_counts())

                st.download_button(
                    'Download predictions as CSV',
                    results.to_csv(index=False).encode('utf-8'),
                    file_name='gbv_form_predictions.csv',
                    mime='text/csv',
                )
