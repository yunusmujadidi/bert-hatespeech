import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
    model = BertForSequenceClassification.from_pretrained('blacklotusid/hsbert')
    return tokenizer,model


tokenizer,model = get_model()

np.random.seed(42)
if "cm" not in st.session_state:
    st.session_state.cm = {}
if "acc" not in st.session_state:
    st.session_state.acc = {}
judul = "Optimasi Algoritma Backpropagation untuk Klasifikasi Customer Churn dengan BFPA dan ENN"
st.set_page_config(
    initial_sidebar_state="expanded",
    page_title=judul,
    layout="centered",
    page_icon="random",
)

col1, col2, col3 = st.columns(3)
with col2:
    st.image("images/unnes.png")
st.markdown(
    "<h1 style='text-align: center'>Skripsi</h1>", unsafe_allow_html=True
)
st.markdown(
    f"<h1 style='text-align: center'>{judul}</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='text-align: center'>Disusun oleh: Hatif Muhammad Fawwaz</h2>",
    unsafe_allow_html=True,
)

user_input = st.text_area('Masukkan text yang ingin di prediksi')
button = st.button("Prediksi")

