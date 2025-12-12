# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PSX 90-day Predictor")

# ================================
#   Extract models.zip (only once)
# ================================
if os.path.exists("models.zip") and not os.path.exists("models_extracted"):
    with zipfile.ZipFile("models.zip", "r") as z:
        z.extractall("models_extracted")

MODEL_DIR = "models_extracted"

# ================================
#   Helper Functions
# ================================
def load_models(ticker):
    """Load RF and XGB models from extracted folder."""
    rf_path = os.path.join(MODEL_DIR, f"{ticker}_RF.pkl")
    xgb_path = os.path.join(MODEL_DIR, f"{ticker}_XGB.pkl")

    rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
    xgb = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    return rf, xgb


def load_preprocessed_csv(ticker):
    """Load preprocessed CSV from repo."""
    fname = f"{ticker}_Preprocessed.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        return df
    st.error(f"Preprocessed file not found: {fname}")
    st.stop()


def construct_feature_row(base_row, edits, user_inputs, feature_cols):
    """Create single prediction row with proper features."""
    row = base_row.copy()

    for k, v in edits.items():
        if k in row.index:
            row[k] = v

    for k, v in user_inputs.items():
        if k in row.index:
            row[k] = v

    row = row.apply(pd.to_numeric, errors="coerce")
    return row[feature_cols].to_frame().T


# ================================
#           UI Header
# ================================
st.title("PSX — 90-Day Stock Price Predictor (Random Forest + XGBoost)")
st.markdown(
    "This tool predicts **90-day forward stock prices** using hybrid technical, "
    "fundamental and market-wide features."
)
st.info("The model is trained on data up to **28 November 2025**.")

st.markdown("**VERSION CHECK:** UI-CLEAN (Date Hidden, Stable Latest Snapshot)")

# ================================
#       Select Stock
# ================================
stock_list = ["EFERT", "MLCF", "MARI", "SAZEW", "TOMCL", "NBP"]
selected = st.sidebar.selectbox("Choose Stock", stock_list)

rf_model, xgb_model = load_models(selected)

if rf_model is None or xgb_model is None:
    st.error("❌ Could not load models for this stock. Ensure .pkl files exist.")
    st.stop()

# ================================
#     Load Preprocessed Data
# ================================
df = load_preprocessed_csv(selected)

# Clean all numbers EXCEPT Date (which we will hide anyway)
df = df.replace({",": ""}, regex=True)
for col in df.columns:
    if col != "Date":
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.fillna(method="ffill").fillna(method="bfill")

# Identify feature columns
feature_cols = [c for c in df.columns if c not in ["Date", "Target_90d", "Return_90d"]]

# ================================
#  SHOW LATEST ROW (DATE HIDDEN)
# ================================
latest_row = df.iloc[-1].copy()
latest_display = latest_row.drop(labels=["Date"])  # HIDE DATE

st.subheader("Latest Available Training Snapshot (Date Hidden)")
st.table(latest_display.to_frame(name="Value"))

# Optional sparkline (last 30 days)
if "Price" in df.columns:
    st.markdown("#### Last 30-day Price Sparkline")
    last_30 = df["Price"].tail(30).tolist()
    st.line_chart(last_30)

# ================================
# FUNDAMENTALS SECTION
# ================================
st.subheader("Fundamental Ratios (Editable)")

fund_cols = ["PE", "EPS", "PB", "DividendYield", "DebtToEquity"]
if selected == "NBP":
    fund_cols.append("RoAA")
else:
    fund_cols.append("NetIncomeMargin")

edit_funds = st.checkbox("Edit Fundamental Ratios?", value=False)

fund_edits = {}
for f in fund_cols:
    default_val = float(latest_row.get(f, 0.0))
    if edit_funds:
        new_val = st.number_input(f"{f}", value=default_val, format="%.6f")
    else:
        st.write(f"**{f}:** {default_val}")
        new_val = default_val
    fund_edits[f] = new_val

# ================================
# DAILY USER INPUT SECTION
# ================================
st.subheader("Daily Market Inputs")

daily_inputs = {}
required_daily = ["Price", "Volume", "RSI14", "KSE_Close", "KSE_Volume"]

for c in required_daily:
    if c in feature_cols:
        default_val = float(latest_row[c])
        if c in ["Volume", "KSE_Volume"]:
            v = st.number_input(f"{c} (no commas)", value=int(default_val), step=1)
        else:
            v = st.number_input(f"{c}", value=default_val, format="%.4f")
        daily_inputs[c] = v

# ================================
#       BUILD FINAL ROW
# ================================
final_row = construct_feature_row(latest_row, fund_edits, daily_inputs, feature_cols)

st.markdown("### Features Sent to Model")
st.dataframe(final_row)

# ================================
#       PREDICT BUTTON
# ================================
if st.button("Predict 90-Day Price"):
    try:
        pred_rf = rf_model.predict(final_row)[0]
        pred_xgb = xgb_model.predict(final_row)[0]
        current_price = final_row.iloc[0]["Price"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Random Forest Predicted Price (90d)", f"{pred_rf:.2f} PKR")
            st.write(f"Expected Return: {(pred_rf-current_price)/current_price*100:.2f}%")

        with col2:
            st.metric("XGBoost Predicted Price (90d)", f"{pred_xgb:.2f} PKR")
            st.write(f"Expected Return: {(pred_xgb-current_price)/current_price*100:.2f}%")

        # Mini Graph
        st.subheader("Prediction Comparison")
        fig, ax = plt.subplots()
        ax.plot(["Today", "RF", "XGB"], [current_price, pred_rf, pred_xgb], marker="o")
        ax.set_ylabel("Price (PKR)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
st.write("Developed for CAIP Final Project • PSX 90-Day Prediction Tool")


