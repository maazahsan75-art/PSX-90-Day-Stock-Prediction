import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PSX 90-Day Predictor")

# -----------------------------------------------------------------------------
# EXTRACT MODELS ONLY ONCE
# -----------------------------------------------------------------------------
if os.path.exists("models.zip") and not os.path.exists("models_extracted"):
    with zipfile.ZipFile("models.zip", "r") as z:
        z.extractall("models_extracted")

MODELS_DIR = "models_extracted"
DATA_DIR = "."
st.write("Files in repo root:", os.listdir("."))
st.write("Files in models_extracted:", os.listdir("models_extracted") if os.path.exists("models_extracted") else "Not extracted yet")

# -----------------------------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------------------------
def load_models(ticker):
    rf_path = os.path.join(MODELS_DIR, f"{ticker}_RF.pkl")
    xgb_path = os.path.join(MODELS_DIR, f"{ticker}_XGB.pkl")

    rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
    xgb = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    return rf, xgb

# -----------------------------------------------------------------------------
# LOAD PREPROCESSED CSV
# -----------------------------------------------------------------------------
def load_preprocessed(ticker):
    fname = os.path.join(DATA_DIR, f"{ticker}_Preprocessed.csv")
    if os.path.exists(fname):
        return pd.read_csv(fname)
    return None

# -----------------------------------------------------------------------------
# CONSTRUCT FEATURE ROW FOR PREDICTION
# -----------------------------------------------------------------------------
def construct_feature_row(base_row, fund_updates, user_inputs, feature_cols):
    row = base_row.copy()

    # apply fundamentals
    for k, v in fund_updates.items():
        if k in row.index:
            row[k] = v

    # apply user daily inputs
    for k, v in user_inputs.items():
        if k in row.index:
            row[k] = v

    row = row.apply(pd.to_numeric, errors='coerce')
    row = row[feature_cols]  # enforce correct column order

    return row.to_frame().T


# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------
st.title("📈 PSX — 90-Day Price Predictor (Random Forest & XGBoost)")
st.write("Select a stock, optionally modify fundamentals, enter today's market inputs, and view the 90-day forecast.")

stock_list = ["EFERT", "MLCF", "MARI", "SAZEW", "TOMCL", "NBP"]
selected = st.sidebar.selectbox("Select Stock", stock_list)

# Load model files
rf_model, xgb_model = load_models(selected)

if rf_model is None or xgb_model is None:
    st.error("❌ Required model files not found in models_extracted/. Upload models.zip correctly.")
    st.stop()

# Load dataset
df = load_preprocessed(selected)
if df is None:
    st.error(f"❌ Preprocessed file {selected}_Preprocessed.csv missing in repository.")
    st.stop()

# Clean dataset
df = df.replace({',': ''}, regex=True)
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.fillna(method="ffill").fillna(method="bfill")

st.markdown("### 📌 Latest Available Row")
st.dataframe(df.tail(1))

# Determine feature columns
feature_cols = [c for c in df.columns if c not in ("Date", "Target_90d", "Return_90d")]

# Last row as base
base_row = df.iloc[-1][feature_cols]

# -----------------------------------------------------------------------------
# FUNDAMENTALS SECTION
# -----------------------------------------------------------------------------
st.markdown("## 🧮 Fundamentals (auto-loaded)")

if selected == "NBP":
    fund_cols = ["PE", "EPS", "PB", "DividendYield", "DebtToEquity", "RoAA"]
else:
    fund_cols = ["PE", "EPS", "PB", "DividendYield", "DebtToEquity", "NetIncomeMargin"]

edit_funds = st.checkbox("Edit Fundamental Ratios?", value=False)

fund_updates = {}

for f in fund_cols:
    default_val = base_row.get(f, 0.0)
    if pd.isna(default_val):
        default_val = 0.0

    if edit_funds:
        fund_updates[f] = st.number_input(f"{f}", value=float(default_val), format="%.6f")
    else:
        st.write(f"**{f}:** {default_val}")
        fund_updates[f] = float(default_val)


# -----------------------------------------------------------------------------
# DAILY INPUT FIELD SECTION
# -----------------------------------------------------------------------------
st.markdown("## 📤 Daily Inputs (Today's Market Data)")

user_inputs = {}
daily_fields = ["Price", "Volume", "RSI14", "KSE_Close", "KSE_Volume"]

for d in daily_fields:
    if d in feature_cols:
        default_val = base_row.get(d, 0.0)
        if pd.isna(default_val):
            default_val = 0.0

        if d in ["Volume", "KSE_Volume"]:
            user_inputs[d] = st.number_input(d, value=int(default_val), step=1)
        else:
            user_inputs[d] = st.number_input(d, value=float(default_val), format="%.6f")


# -----------------------------------------------------------------------------
# BUILD FEATURE VECTOR
# -----------------------------------------------------------------------------
final_row = construct_feature_row(base_row, fund_updates, user_inputs, feature_cols)

st.markdown("### 🔍 Feature Vector Sent to Model")
st.dataframe(final_row.T)

# -----------------------------------------------------------------------------
# PREDICT BUTTON
# -----------------------------------------------------------------------------
if st.button("🚀 Predict 90-Day Price"):
    final_numeric = final_row.apply(pd.to_numeric, errors='coerce')

    # RF prediction
    pred_rf = rf_model.predict(final_numeric)[0]

    # XGB prediction
    pred_xgb = xgb_model.predict(final_numeric)[0]

    current_price = final_numeric.iloc[0]["Price"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random Forest Prediction (90d)", f"{pred_rf:.2f} PKR")
        st.write(f"Return: {(pred_rf - current_price) / current_price * 100:.2f}%")

    with col2:
        st.metric("XGBoost Prediction (90d)", f"{pred_xgb:.2f} PKR")
        st.write(f"Return: {(pred_xgb - current_price) / current_price * 100:.2f}%")

    # Quick mini plot
    st.markdown("### 📊 Prediction Comparison")
    fig, ax = plt.subplots()
    ax.plot(["Today", "RF", "XGB"], [current_price, pred_rf, pred_xgb], marker="o")
    ax.set_ylabel("Price (PKR)")
    ax.grid(True)
    st.pyplot(fig)


# -----------------------------------------------------------------------------
# OPTIONAL METRICS
# -----------------------------------------------------------------------------
if st.checkbox("Show Quick Model Metrics"):
    from sklearn.model_selection import train_test_split

    X = df[feature_cols]
    y = df["Target_90d"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pred_rf_test = rf_model.predict(Xte)
    pred_xgb_test = xgb_model.predict(Xte)

    rmse_rf = np.sqrt(((yte - pred_rf_test) ** 2).mean())
    rmse_xgb = np.sqrt(((yte - pred_xgb_test) ** 2).mean())

    st.write(f"RF RMSE: {rmse_rf:.4f}")
    st.write(f"XGB RMSE: {rmse_xgb:.4f}")

st.write("---")
st.write("App built for CAIP Project — Models auto-loaded from models.zip.")
