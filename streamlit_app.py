import streamlit as st
import pandas as pd
import numpy as np

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Dashboard Prediksi Popularitas Film",
    page_icon="🎬",
    layout="wide"
)

# ==========================
# CSS
# ==========================
st.markdown("""
<style>
.stApp{
    background-color:#0E1117;
    color:white;
}

div[data-testid="metric-container"]{
    background:#1c2333;
    border-radius:10px;
    padding:18px;
    border:1px solid #444;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Judul
# ==========================
st.title("🎬 Dashboard Prediksi Popularitas Film")
st.markdown("---")

# ==========================
# Load Dataset
# ==========================
try:
    df = pd.read_csv("mymoviedb (1) (1).csv")
except FileNotFoundError:
    st.error("Dataset 'mymoviedb (1) (1).csv' tidak ditemukan.")
    st.stop()

# ==========================
# Preprocessing
# ==========================
df["Release_Date"] = pd.to_datetime(df["Release_Date"])

df["Year"] = df["Release_Date"].dt.year

# ==========================
# Ringkasan Dataset
# ==========================
st.header("📊 Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Jumlah Film", len(df))
col2.metric("Rata-rata Rating", round(df["Vote_Average"].mean(),2))
col3.metric("Jumlah Genre", df["Genre"].nunique())
col4.metric("Jumlah Bahasa", df["Original_Language"].nunique())

# ==========================
# Dataset
# ==========================
st.header("📄 Dataset Film")

st.dataframe(df, use_container_width=True)
