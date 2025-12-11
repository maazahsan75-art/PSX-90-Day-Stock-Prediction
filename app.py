import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PSX 90-Day Stock Predictor")


# =====================================================
# 1 — Extract models.zip (only once)
# =====================================================
if os.path.exists("models.zip") and not os.path.exists("models_extracted"):
    with zipfile.ZipFile("models.zip", "r") as z:
        z.extractall("models_extracted")


# =====================================================
# 2 — Helper: Load models
# =====================================================
def load_models(ticker: str):
    rf_path = os.path.join("models_extracted", f"{ticker}_RF.pkl")
    xgb_path = os.path.join("models_extracted", f"{ticker}_XGB.pkl")

    rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
    xgb_model = joblib.load(xgb_path) if os.path.exists(xgb_path) else None

    return rf_model, xgb_model


# =====================================================
# 3 — Helper: Load preprocessed CSV
# =====================================================
def get_default_df(ticker: str):
    filename = f"{ticker}_Preprocessed.csv"

    # Try root directory
    if os.path.exists(filename):
        return pd.read_csv(filename)

    # Try capitalization variations
    filename2 = f"{ticker}_Preprocessed.CSV"
    if os.path.exists(filename2):
        return pd.read_csv(filename2)

    return None


# =====================================================
# 4 — Combine user inputs + fundamentals into one row
# =====================================================
def construct_feature_row(base_row, edits, user_inputs, feature_cols):
    row = base_row.copy()

    # Apply fundamental edits
    for key, val in edits.items():
        if key in row.index:
            row[key] = val

    # Apply market inputs
    for key, val in user_inputs.items():
        if key in row.index:
            row[key] = val

    row = row.apply(pd.to_numeric, errors="coerce")
    row = row[feature_cols]

    return row.to_frame().T


# =====================================================
# 5 — UI Layout
# =====================================================
st.title("📈 PSX — 90-Day Price Predictor (Random Forest + XGBoost)")

stock_list = ["EFERT", "MLCF", "MARI", "SAZEW", "TOMCL", "NBP"]
selected = st.sidebar.selectbox("Select Stock", stock_list)

# Load models
rf_model, xgb_model = load_models(selected)

if rf_model is None or xgb_model is None:
    st.error("❌ Models not found in models_extracted/ — fix your models.zip upload.")
    st.stop()

# Load CSV
df = get_default_df(selected)
if df is None:
    st.error("❌ Preprocessed CSV NOT found. Upload *_Preprocessed.csv* to repo root.")
    st.stop()

# Cleanup numeric data
df = df.replace({",": ""}, regex=True)
for col in df.columns:
    if col != "Date":
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.fillna(method="ffill").fillna(method="bfill")

st.subheader("📌 Latest Available Data (Auto-Filled Defaults)")
st.dataframe(df.tail(1))

# Training feature columns (exclude target columns)
feature_cols = [c for c in df.columns if c not in ["Date", "Target_90d", "Return_90d"]]
base_row = df.iloc[-1][feature_cols]


# =====================================================
# FUNDAMENTALS
# =====================================================
st.subheader("🏛 Fundamental Ratios")

if selected == "NBP":
    fund_cols = ["PE", "EPS", "PB", "DividendYield", "DebtToEquity", "RoAA"]
else:
    fund_cols = ["PE", "EPS", "PB", "DividendYield", "DebtToEquity", "NetIncomeMargin"]

edit_funds = st.checkbox("Edit Fundamental Values?", value=False)

fund_edits = {}
for col in fund_cols:
    default = float(base_row[col]) if col in base_row.index else 0.0

    if edit_funds:
        fund_edits[col] = st.number_input(col, value=default, format="%.6f")
    else:
        st.write(f"**{col}:** {default}")
        fund_edits[col] = default


# =====================================================
# MARKET INPUTS
# =====================================================
st.subheader("📊 Daily Market Inputs")

user_inputs = {}
market_cols = ["Price", "Volume", "RSI14", "KSE_Close", "KSE_Volume"]

for col in market_cols:
    default = float(base_row[col]) if col in base_row.index else 0.0
    if col in ["Volume", "KSE_Volume"]:
        user_inputs[col] = st.number_input(col, value=int(default), step=1)
    else:
        user_inputs[col] = st.number_input(col, value=float(default), format="%.6f")


# =====================================================
# CONSTRUCT FINAL INPUT ROW
# =====================================================
final_row = construct_feature_row(base_row, fund_edits, user_inputs, feature_cols)

st.subheader("🧩 Final Feature Row Sent to Model")
st.dataframe(final_row.T)


# =====================================================
# PREDICTION BUTTON
# =====================================================
if st.button("🔮 Predict 90-Day Price"):

    pred_rf = rf_model.predict(final_row)[0]
    pred_xgb = xgb_model.predict(final_row)[0]

    current_price = final_row.iloc[0]["Price"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("RF Predicted Price (90d)", f"{pred_rf:.2f} PKR")
        st.write(f"Return: {(pred_rf - current_price) / current_price * 100:.2f}%")

    with col2:
        st.metric("XGB Predicted Price (90d)", f"{pred_xgb:.2f} PKR")
        st.write(f"Return: {(pred_xgb - current_price) / current_price * 100:.2f}%")


    # Mini Chart
    fig, ax = plt.subplots()
    ax.plot(["Today", "RF", "XGB"], [current_price, pred_rf, pred_xgb], marker="o")
    ax.set_ylabel("Price (PKR)")
    ax.grid(True)
    st.pyplot(fig)

    # Optional CSV download
    result = final_row.copy()
    result["Pred_RF_90d"] = pred_rf
    result["Pred_XGB_90d"] = pred_xgb

    st.download_button(
        "Download Prediction Row",
        result.to_csv(index=False),
        file_name=f"{selected}_prediction.csv",
        mime="text/csv",
    )


st.markdown("---")
st.write("Built for CAIP Final Project — PSX Forecasting Tool.")
