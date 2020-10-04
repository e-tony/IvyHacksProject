import pathlib
import plotly.express as px
import streamlit as st
# import seaborn as sns
import numpy as np
import pandas as pd
from gensim.models import fasttext as FT
from sklearn.decomposition import PCA
import requests
import json

headers_1 = {'accept': 'application/json'}
headers_2 = {'accept': 'application/json', 'Content-Type': 'application/json'}

st.sidebar.title("Parameters")

st.title("Visualizer")

# =====================================================================
st.sidebar.title("Visualize in 3D")
# Visualize 3D 
hosp_list = st.sidebar.selectbox("1. Select term list with property A", ['female_terms', 'male_terms'])
response_1 = requests.post(f'http://127.0.0.1:8000/terms/{hosp_list}', headers=headers_1).json()
text_1 = st.sidebar.text_input("Terms 1", ", ".join(response_1["terms"]))
viz_terms_1 = text_1.split(", ")

hosp_list = st.sidebar.selectbox("2. Select term list with property A", ['male_terms', 'female_terms'])
response_2 = requests.post(f'http://127.0.0.1:8000/terms/{hosp_list}', headers=headers_1).json()
text_2 = st.sidebar.text_input("Terms 2", ", ".join(response_2["terms"]))
viz_terms_2 = text_2.split(", ")

hosp_list = st.sidebar.selectbox("3. Select term list for neutral words", ['family', 'career'])
response_3 = requests.post(f'http://127.0.0.1:8000/terms/{hosp_list}', headers=headers_1).json()
text_3 = st.sidebar.text_input("Terms 3", ", ".join(response_3["terms"]))
viz_neutral_terms = text_3.split(", ")

button_viz = st.sidebar.button("Visualize")

data_1 = {"property_terms_1": viz_terms_1, "property_terms_2": viz_terms_2, "neutral_terms": viz_neutral_terms}

if button_viz:
    response_a = requests.post('http://127.0.0.1:8000/get_3D_coordinates/fastText', headers=headers_2, data=data_1)

    st.markdown("FastText Model")
    st.markdown("Word embeddings are vectors of numbers that are used to represent words. FastText (Grave et al., 2018) word embeddings were trained on the text on Common Crawl and Wikipedia, which contain many millions of websites. The model learns vectors for words, which are called word embeddings, by prediction a word according to its context. For example, when looking at a sequence of 5 words, the middle word in the sequence is predicted taking the surrounding words into account.")

    _t1 = response_1["terms"]
    _t1_coords = [response_a["property_terms_1"][t] for t in _t1]
    _t2 = response_2["terms"]
    _t2_coords = [response_a["property_terms_2"][t] for t in _t2]
    _t3 = response_3["terms"]
    _t3_coords = [response_a["neutral_terms"][t] for t in _t3]

    col1 = _t1+_t2+_t3
    col2 = ["1"]*len(_t1)+["2"]*len(_t2)+["3"]*len(_t3)
    x,y,z = [], [], []
    for term, (_x,_y,_z) in zip(_t1+_t2+_t3, _t1_coords+_t2_coords+_t3_coords):
        x.append(_x)
        y.append(_y)
        z.append(_z)

    df = pd.DataFrame.from_dict({"x": x, "y": y, "z": z, "text": col1,"type": col2})

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='type', text='text', width=800, height=800)
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig)

# =====================================================================
st.sidebar.title("Debias Model")
# Debiasing method
hosp_list = st.sidebar.selectbox("4. Select term list with property A", ['family', 'career'])
response_4 = requests.post(f'http://127.0.0.1:8000/terms/{hosp_list}', headers=headers_1).json()
text_4 = st.sidebar.text_input("Terms 4", ", ".join(response_4["terms"]))
deb_terms_1 = text_4.split(", ")

hosp_list = st.sidebar.selectbox("5. Select term list with property B", ['family', 'career'])
response_5 = requests.post(f'http://127.0.0.1:8000/terms/{hosp_list}', headers=headers_1).json()
text_5 = st.sidebar.text_input("Terms 5", ", ".join(response_5["terms"]))
deb_terms_2 = text_5.split(", ")

data_2 = {"terms_1": deb_terms_1, "terms_2": deb_terms_2}

button_deb = st.sidebar.button("Run debiasing")

if button_deb:
    requests.post('http://127.0.0.1:8000/fastText?debiasing_method=GBDD', headers=headers_2, data=data_2)
    response_a = requests.post('http://127.0.0.1:8000/get_3D_coordinates/fastText\ debiased', headers=headers_2, data=data_1)

    st.markdown("GBDD Debiasing")
    st.markdown("The Generalized Bias-Direction Debiasing (GBDD) model by Lauscher et al. (2020), removes a bias from word embeddings. To do this, pairs of opposing terms are created from two lists of terms, each with a property that the other does not have. For example, female and male terms like “mother” and “father”. The bias embedding can be found by collecting all differences between all pairs and applying a mathematical operation to remove it.")
    
    _t1 = response_1["terms"]
    _t1_coords = [response_a["property_terms_1"][t] for t in _t1]
    _t2 = response_2["terms"]
    _t2_coords = [response_a["property_terms_2"][t] for t in _t2]
    _t3 = response_3["terms"]
    _t3_coords = [response_a["neutral_terms"][t] for t in _t3]

    col1 = _t1+_t2+_t3
    col2 = ["1"]*len(_t1)+["2"]*len(_t2)+["3"]*len(_t3)
    x,y,z = [], [], []
    for term, (_x,_y,_z) in zip(_t1+_t2+_t3, _t1_coords+_t2_coords+_t3_coords):
        x.append(_x)
        y.append(_y)
        z.append(_z)

    df = pd.DataFrame.from_dict({"x": x, "y": y, "z": z, "text": col1,"type": col2})

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='type', text='text', width=800, height=800)
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig)

# =====================================================================
st.sidebar.title("Evaluate on Metric")
# Metric lists
hosp_list = st.sidebar.selectbox("6. Select term list with property A", ['family', 'career'])
response_6 = requests.post(f'http://127.0.0.1:8000/terms/{hosp_list}', headers=headers_1).json()
text_6 = st.sidebar.text_input("Terms 6", ", ".join(response_6["terms"]))
metric_terms_1 = text_6.split(", ")

hosp_list = st.sidebar.selectbox("7. Select term list with property B", ['family', 'career'])
response_7 = requests.post(f'http://127.0.0.1:8000/terms/{hosp_list}', headers=headers_1).json()
text_7 = st.sidebar.text_input("Terms 7", ", ".join(response_7["terms"]))
metric_terms_2 = text_7.split(", ")

hosp_list = st.sidebar.selectbox("8. Select attribute list with property C", ['family', 'career'])
response_8 = requests.post(f'http://127.0.0.1:8000/terms/{hosp_list}', headers=headers_1).json()
text_8 = st.sidebar.text_input("Terms 8", ", ".join(response_8["terms"]))
metric_attr_1 = text_8.split(", ")

hosp_list = st.sidebar.selectbox("9. Select attribute list with property D", ['family', 'career'])
response_9 = requests.post(f'http://127.0.0.1:8000/terms/{hosp_list}', headers=headers_1).json()
text_9 = st.sidebar.text_input("Terms 9", ", ".join(response_9["terms"]))
metric_attr_2 = text_9.split(", ")

button_metric = st.sidebar.button("Run metric")

data_3 = {"property_terms_1": metric_terms_1, "property_terms_2": metric_terms_2, "attribute_terms_1": metric_attr_1, "attribute_terms_1": metric_attr_2}

if button_metric:
    response_c = requests.post('http://127.0.0.1:8000/get_metric/fastText?metric_name=WEAT', headers=headers_2, data=data_3)

    st.markdown("WEAT Metric")
    st.markdown("The Word Embedding Association Test (WEAT) (Caliskan et al., 2017) calculates a score that measures how much bias a word embedding contains based on the words used to evaluate it. The choice of words thus affects the outcome. The metric requires two sets of target and attribute terms.")
    
    st.markdown(f"WEAT score: {response_c["result"]}")