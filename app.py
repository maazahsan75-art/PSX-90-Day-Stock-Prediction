# -----------------------------  
#  PSX 90-DAY PRICE PREDICTOR  
#  Final Streamlit Application  
#  With cleaned latest row fix, last updated stamp & sparkline  
# -----------------------------  

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PSX 90-Day Predictor")

st.error("VERSION CHECK: CLEANER LOADER ACTIVE")



# -------------------------------------
# Extract models.zip ONCE
# -------------------------------------
if os.path.exists("models.zip") and not os.path.exists("models_extracted"):
    with zipfile.ZipFile("models.zip", "r") as z:
        z.extractall("models_extracted")

MODELS_PATH = "models_extracted"


# -------------------------------------
# Helper: Load Models
# -------------------------------------
def load_models(ticker):
    rf_path = os.path.join(MODELS_PATH, f"{ticker}_RF.pkl")
    xgb_path = os.path.join(MODELS_PATH, f"{ticker}_XGB.pkl")

    rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
    xgb = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    return rf, xgb


# -------------------------------------
# Helper: Load Cleaned CSV
# -------------------------------------
def load_preprocessed(ticker):
    fname = f"{ticker}_Preprocessed.csv"

    if not os.path.exists(fname):
        st.error(f"❌ {fname} not found in repo!")
        return None

    # Load raw CSV strictly as text (prevents wrong type inference)
    df = pd.read_csv(fname, dtype=str, keep_default_na=False)

    # ---------------------------------------
    # 🔥 1) REMOVE ALL EMPTY OR WHITESPACE ROWS
    # ---------------------------------------
    df = df[~df.apply(lambda r: r.astype(str).str.strip().eq("").all(), axis=1)]

    # ---------------------------------------
    # 🔥 2) CLEAN COMMAS, SPACES, NON-NUMERIC GARBAGE
    # ---------------------------------------
    df = df.replace({',': ''}, regex=True)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # ---------------------------------------
    # 🔥 3) PARSE DATE USING STRICT MULTI-FORMAT
    # ---------------------------------------
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        infer_datetime_format=True
    )

    # Remove rows where Date failed to parse
    df = df.dropna(subset=["Date"])

    # ---------------------------------------
    # 🔥 4) CONVERT NUMERIC COLUMNS PROPERLY
    # ---------------------------------------
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------------------------------
    # 🔥 5) DROP ROWS WHERE ALL NUMERIC COLUMNS ARE NaN
    # ---------------------------------------
    numeric_cols = [c for c in df.columns if c != "Date"]
    df = df.dropna(subset=numeric_cols, how="all")

    # ---------------------------------------
    # 🔥 6) SORT BY DATE ASCENDING & RESET
    # ---------------------------------------
    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------
    # 🔥 7) FORWARD + BACKWARD FILL FOR GAPS
    # ---------------------------------------
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df

# -------------------------------------
# Helper: Build Feature Vector
# -------------------------------------
def construct_feature_row(base_row, edits, user_inputs, feature_cols):
    row = base_row.copy()

    for k, v in edits.items():
        if k in row.index:
            row[k] = v

    for k, v in user_inputs.items():
        if k in row.index:
            row[k] = v

    row = row.apply(pd.to_numeric, errors="coerce")
    row = row[feature_cols]

    return row.to_frame().T


# -------------------------------------
# UI Layout
# -------------------------------------
st.title("📈 PSX — 90-Day Price Predictor (Random Forest & XGBoost)")
st.write("Select a stock, review fundamentals, enter today's inputs, and get predictions.")

stock_list = ["EFERT", "MLCF", "MARI", "SAZEW", "TOMCL", "NBP"]
selected = st.sidebar.selectbox("Select Stock", stock_list)

# Load Models
rf_model, xgb_model = load_models(selected)

if rf_model is None or xgb_model is None:
    st.error("❌ Required model files missing in *models_extracted/*")
    st.stop()

# Load Data
df = load_preprocessed(selected)
if df is None:
    st.stop()

# ------------------------------------------
# FIX: Force detection of TRUE latest date
# ------------------------------------------
max_date = df["Date"].max()
latest_row = df[df["Date"] == max_date].tail(1)

latest_date = max_date.strftime("%Y-%m-%d")

st.markdown("## 📌 Latest Available Training Data Snapshot")
st.write(f"**Last Updated:** {latest_date}")

# Show the cleaned row
st.dataframe(latest_row, height=150)

# Sparkline (last 30 closing prices)
with st.expander("📉 Show last 30-day price sparkline"):
    fig, ax = plt.subplots(figsize=(5, 1.5))
    price_data = df["Price"].tail(30).values
    ax.plot(price_data, linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)


# -------------------------------------
# Fundamentals Section
# -------------------------------------
st.markdown("## 🧮 Fundamentals (auto-filled)")

feature_cols = [c for c in df.columns if c not in ("Date", "Target_90d", "Return_90d")]
base_row = df.iloc[-1][feature_cols]

fund_cols = ["PE", "EPS", "PB", "DividendYield", "DebtToEquity"]
fund_cols.append("RoAA" if selected == "NBP" else "NetIncomeMargin")

edit_funds = st.checkbox("Edit fundamentals?", value=False)

fund_edits = {}
for f in fund_cols:
    default_val = float(base_row.get(f, 0.0))
    if edit_funds:
        new_val = st.number_input(f"{f}:", value=default_val, format="%.6f")
    else:
        st.write(f"**{f}:** {default_val}")
        new_val = default_val
    fund_edits[f] = new_val


# -------------------------------------
# User Inputs
# -------------------------------------
st.markdown("## 📊 Daily Inputs")

user_inputs = {}
for field in ["Price", "Volume", "RSI14", "KSE_Close", "KSE_Volume"]:
    if field in base_row.index:
        default = float(base_row[field])
        if field in ["Volume", "KSE_Volume"]:
            val = st.number_input(f"{field}:", value=int(default))
        else:
            val = st.number_input(f"{field}:", value=float(default), format="%.6f")
        user_inputs[field] = val


# -------------------------------------
# Create Row & Predict
# -------------------------------------
final_row = construct_feature_row(base_row, fund_edits, user_inputs, feature_cols)

st.markdown("### 🔍 Features Sent to Model")
st.dataframe(final_row.T)

if st.button("🚀 Predict 90-Day Price"):
    try:
        pred_rf = rf_model.predict(final_row)[0]
        pred_xgb = xgb_model.predict(final_row)[0]
        current_price = final_row.iloc[0]["Price"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Random Forest Prediction (90d)", f"{pred_rf:.2f} PKR",
                      f"{((pred_rf - current_price)/current_price)*100:.2f}%")

        with col2:
            st.metric("XGBoost Prediction (90d)", f"{pred_xgb:.2f} PKR",
                      f"{((pred_xgb - current_price)/current_price)*100:.2f}%")

        # Tiny comparison chart
        fig2, ax2 = plt.subplots()
        ax2.plot([0,1,2], [current_price, pred_rf, pred_xgb], marker="o")
        ax2.set_xticks([0,1,2])
        ax2.set_xticklabels(["Today", "RF", "XGB"])
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
