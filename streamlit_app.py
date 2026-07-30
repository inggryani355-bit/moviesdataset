import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Dashboard Prediksi Popularitas Film",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp{
    background-color:#0E1117;
    color:white;
}

div[data-testid="metric-container"]{
    background:#1d2330;
    padding:20px;
    border-radius:12px;
    border:1px solid #333;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Dashboard Prediksi Popularitas Film")
st.markdown("---")

# =====================
# LOAD DATA
# =====================

df = pd.read_csv("mymoviedb.csv")

df["Release_Date"] = pd.to_datetime(df["Release_Date"])

df["Year"] = df["Release_Date"].dt.year

# =====================
# RINGKASAN
# =====================

st.header("1. Ringkasan Dataset")

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Jumlah Film",
    len(df)
)

c2.metric(
    "Rata-rata Rating",
    round(df["Vote_Average"].mean(),2)
)

c3.metric(
    "Total Genre",
    df["Genre"].nunique()
)

c4.metric(
    "Bahasa",
    df["Original_Language"].nunique()
)

st.dataframe(df)
