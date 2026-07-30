import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
    df = pd.read_csv(
        "mymoviedb (1) (1).csv",
        engine="python",
        on_bad_lines="skip"
    )
except FileNotFoundError:
    st.error("Dataset 'mymoviedb (1) (1).csv' tidak ditemukan.")
    st.stop()

# ==========================
# Preprocessing
# ==========================
df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df = df.dropna(subset=["Release_Date"])
df["Year"] = df["Release_Date"].dt.year

numeric_cols = ["Popularity", "Vote_Count", "Vote_Average"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)

# ==========================
# Ringkasan Dataset
# ==========================
st.header("📊 Ringkasan Dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Film", len(df))
col2.metric("Rata-rata Rating", round(df["Vote_Average"].mean(), 2))
col3.metric("Jumlah Genre", df["Genre"].nunique())
col4.metric("Jumlah Bahasa", df["Original_Language"].nunique())

# ==========================
# Dataset
# ==========================
st.header("📄 Dataset Film")
st.dataframe(df, use_container_width=True)

st.markdown("---")

# ===========================
# Distribusi Vote Average
# ===========================
st.header("📈 Distribusi Rating Film (Vote Average)")

fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df["Vote_Average"], bins=20, kde=True, ax=ax1, color="skyblue")
ax1.set_xlabel("Vote_Average")
ax1.set_ylabel("Count")
st.pyplot(fig1)

st.markdown("---")

# ===========================
# Top 15 Genre
# ===========================
st.header("🎭 Top 15 Genre Film yang Paling Sering Diproduksi")

genre_series = df["Genre"].dropna().str.split(", ").explode()
top_genres = genre_series.value_counts().head(15)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(
    x=top_genres.values,
    y=top_genres.index,
    palette="magma",
    ax=ax2
)
ax2.set_xlabel("Jumlah Kemunculan dalam Dataset")
ax2.set_ylabel("")
ax2.set_title("Top 15 Genre Film yang Paling Sering Diproduksi")
st.pyplot(fig2)

st.markdown("---")

# ===========================
# Tren Jumlah Rilis Film per Tahun
# ===========================
st.header("📅 Tren Jumlah Perilisan Film (20 Tahun Terakhir)")

tahun_terakhir = df["Year"].max()
df_20tahun = df[df["Year"] >= tahun_terakhir - 19]
trend = df_20tahun.groupby("Year").size()

fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.plot(trend.index, trend.values, marker="o", color="red")
ax3.set_xlabel("Tahun Rilis")
ax3.set_ylabel("Jumlah Film")
ax3.set_title("Tren Jumlah Perilisan Film (20 Tahun Terakhir)")
plt.xticks(rotation=45)
st.pyplot(fig3)

st.markdown("---")

# ===========================
# Korelasi Antar Kolom
# ===========================
st.header("🔗 Korelasi Antar Kolom")

corr_cols = ["Popularity", "Vote_Count", "Vote_Average", "Year"]
corr_matrix = df[corr_cols].corr()

fig4, ax4 = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
ax4.set_title("Korelasi Antar Kolom")
st.pyplot(fig4)

st.markdown("---")

# ===========================
# Model Regresi Linear
# ===========================
st.header("🤖 Model Prediksi Popularitas Film")

X = df[["Vote_Count", "Vote_Average", "Year"]]
Y = df["Popularity"]

model = LinearRegression()
model.fit(X, Y)

st.success("Model Linear Regression berhasil dilatih menggunakan fitur: Vote_Count, Vote_Average, dan Year.")

# ===========================
# Actual vs Predicted
# ===========================
st.subheader("🎯 Actual vs Predicted Popularity")

y_pred = model.predict(X)

fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.scatter(Y, y_pred, alpha=0.6)
ax5.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color="red")
ax5.set_xlabel("Actual")
ax5.set_ylabel("Predicted")
ax5.set_title("Actual vs Predicted Popularity")
st.pyplot(fig5)

# ===========================
# Evaluasi Model
# ===========================
r2 = r2_score(Y, y_pred)
mae = mean_absolute_error(Y, y_pred)
rmse = np.sqrt(mean_squared_error(Y, y_pred))

st.subheader("📈 Evaluasi Model Regresi")

eval_col1, eval_col2, eval_col3 = st.columns(3)
eval_col1.metric("R² Score", round(r2, 3))
eval_col2.metric("MAE", round(mae, 2))
eval_col3.metric("RMSE", round(rmse, 2))

st.caption(
    "R² mendekati 1 berarti model semakin baik menjelaskan variasi data. "
    "MAE dan RMSE menunjukkan rata-rata besar kesalahan prediksi (semakin kecil semakin baik)."
)
