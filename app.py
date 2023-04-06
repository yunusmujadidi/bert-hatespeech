import streamlit as st
import numpy as np

np.random.seed(42)
if "cm" not in st.session_state:
    st.session_state.cm = {}
if "acc" not in st.session_state:
    st.session_state.acc = {}
judul = "DETEKSI HATE SPEECH BERBAHASA INDONESIA PADA MEDIA SOSIAL TWITTER MENGGUNAKAN BIDIRECTIONAL ENCODER REPRESENTATION FROM TRANSFORMERS (BERT)"
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
    "<h2 style='text-align: center'>Disusun oleh: Ahmad Yunus Mujadidi</h2>",
    unsafe_allow_html=True,
)

