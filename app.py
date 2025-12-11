# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="PSX 90-day Predictor")

# ------- Helper functions -------
def safe_numeric_series(s):
    s = s.replace({',': ''}, regex=True)
    return pd.to_numeric(s, errors='coerce')

def load_models(ticker, models_path="."):
    """Load RF and XGB models for a given ticker from models_path"""
    rf_path = os.path.join(models_path, f"{ticker}_RF.pkl")
    xgb_path = os.path.join(models_path, f"{ticker}_XGB.pkl")
    rf = None
    xgb = None
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
    if os.path.exists(xgb_path):
        xgb = joblib.load(xgb_path)
    return rf, xgb

def get_default_df(ticker, data_path="."):
    """Try to load preprocessed CSV for ticker from either repo or uploaded files folder"""
    candidates = [
        os.path.join(data_path, f"{ticker}_Preprocessed.csv"),
        os.path.join(data_path, f"{ticker}_Preprocessed.CSV"),
        os.path.join(".", f"{ticker}_Preprocessed.csv"),
        os.path.join(".", f"{ticker}_Preprocessed.CSV")
    ]
    for c in candidates:
        if os.path.exists(c):
            df = pd.read_csv(c)
            return df
    return None

def construct_feature_row(base_row, edits, user_inputs, feature_cols):
    """Construct a single-row DataFrame matching training features order."""
    row = base_row.copy()
    # apply fundamental edits (edits is dict)
    for k, v in edits.items():
        if k in row.index:
            row[k] = v
    # apply user inputs
    for k, v in user_inputs.items():
        if k in row.index:
            row[k] = v
    # ensure numeric dtypes
    row = row.apply(pd.to_numeric, errors='coerce')
    # keep only the feature columns used during training
    row = row[feature_cols]
    return row.to_frame().T

# ------- UI layout -------
st.title("PSX — 90-Day Price Predictor (Random Forest & XGBoost)")
st.write("Select a stock, optionally edit fundamentals, enter today's market inputs, and get 90-day price forecasts.")

# Sidebar: model/data location & stock selection
st.sidebar.header("App settings / model files")
models_path = st.sidebar.text_input("Models folder (relative to app.py)", value=".", help="Place the <TICKER>_RF.pkl and <TICKER>_XGB.pkl files here.")
data_path = st.sidebar.text_input("Preprocessed CSV folder (relative to app.py)", value=".", help="Place *Preprocessed.csv files here (optional).")

stock_list = ["EFERT", "MLCF", "MARI", "SAZEW", "TOMCL", "NBP"]
selected = st.sidebar.selectbox("Choose stock", stock_list)

# Load models
rf_model, xgb_model = load_models(selected, models_path=models_path)
if rf_model is None or xgb_model is None:
    st.sidebar.warning("RF or XGB model not found for selected stock in models folder. Upload models or check path.")
else:
    st.sidebar.success("Loaded RF & XGB models.")

# Option to upload preprocessed CSVs (one or many)
st.sidebar.markdown("---")
uploaded_csv = st.sidebar.file_uploader("(Optional) Upload this stock's preprocessed CSV (select file if not in repo)", type=["csv"], accept_multiple_files=False)
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
else:
    df = get_default_df(selected, data_path=data_path)
    if df is None:
        st.warning("Preprocessed CSV not found in repo or upload. Please upload the preprocessed CSV for selected stock.")
        st.stop()

# Basic cleaning
df = df.replace({',': ''}, regex=True)
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.fillna(method="ffill").fillna(method="bfill")

st.markdown("### Latest available data snapshot (last row used for defaults)")
st.dataframe(df.tail(1), height=160)

# Feature columns used for training (drop Date, Target_90d, Return_90d)
feature_cols = [c for c in df.columns if c not in ("Date", "Target_90d", "Return_90d")]

# Display fundamentals (6 ratios)
st.markdown("## Fundamentals (latest values)")
# For NBP use RoAA instead of NetIncomeMargin
fund_cols = ["PE", "EPS", "PB", "DividendYield", "DebtToEquity"]
if selected == "NBP":
    fund_cols.append("RoAA")
else:
    fund_cols.append("NetIncomeMargin")

# Create base row for defaults
base_row = df.iloc[-1][feature_cols]

# Show fundamentals and allow edit toggle
st.write("Latest fundamentals (auto-filled from last available data). You can edit them if you wish.")
edit_funds = st.checkbox("Edit fundamental ratios manually?", value=False)

fund_edits = {}
fund_values_cols = []
for f in fund_cols:
    if f in base_row.index:
        default_val = float(base_row[f]) if not pd.isna(base_row[f]) else 0.0
    else:
        default_val = 0.0
    if edit_funds:
        new_val = st.number_input(f"{f}", value=default_val, format="%.6f")
    else:
        st.write(f"**{f}:** {default_val}")
        new_val = default_val
    fund_edits[f] = new_val
    fund_values_cols.append(f)

# Now user inputs for daily-changing fields
st.markdown("## Daily Inputs (enter today's market values)")
user_inputs = {}
for c in ["Price", "Volume", "RSI14", "KSE_Close", "KSE_Volume"]:
    if c in feature_cols:
        default_val = float(base_row[c]) if not pd.isna(base_row[c]) else 0.0
        if c in ["Volume", "KSE_Volume"]:
            v = st.number_input(f"{c} (no commas)", value=int(default_val), step=1)
        else:
            v = st.number_input(f"{c}", value=float(default_val), format="%.6f")
        user_inputs[c] = v

# Build final feature row
final_row = construct_feature_row(base_row, fund_edits, user_inputs, feature_cols)

st.markdown("### Features sent to model (first row shown)")
st.dataframe(final_row.T)

# Predict button
if st.button("Predict 90-day Price (Random Forest & XGBoost)"):
    if rf_model is None or xgb_model is None:
        st.error("Models not loaded. Ensure .pkl files are present in models folder or uploaded in repo.")
    else:
        # Ensure numeric types for XGBoost
        final_row = final_row.apply(pd.to_numeric, errors='coerce')
        # RF predict
        try:
            pred_rf = rf_model.predict(final_row)[0]
        except Exception as e:
            st.error(f"RandomForest predict error: {e}")
            pred_rf = None
        # XGB predict
        try:
            pred_xgb = xgb_model.predict(final_row)[0]
        except Exception as e:
            st.error(f"XGBoost predict error: {e}")
            pred_xgb = None

        # Current price for return calc
        current_price = final_row.iloc[0]["Price"] if "Price" in final_row.columns else None

        col1, col2 = st.columns(2)
        with col1:
            if pred_rf is not None:
                st.metric(label="Random Forest → Predicted Price (90d)", value=f"{pred_rf:.2f} PKR")
                if current_price:
                    st.write(f"Expected Return (RF): {((pred_rf-current_price)/current_price*100):.2f}%")
        with col2:
            if pred_xgb is not None:
                st.metric(label="XGBoost → Predicted Price (90d)", value=f"{pred_xgb:.2f} PKR")
                if current_price:
                    st.write(f"Expected Return (XGB): {((pred_xgb-current_price)/current_price*100):.2f}%")

        # Quick mini-plot
        st.markdown("#### Local comparison graph")
        fig, ax = plt.subplots()
        ax.plot([0,1,2], [current_price, pred_rf, pred_xgb], marker='o')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(["Today", "RF (90d)", "XGB (90d)"])
        ax.set_ylabel("Price (PKR)")
        ax.grid(True)
        st.pyplot(fig)

        # Allow download of single-row CSV for record
        result_df = final_row.copy()
        result_df["Pred_RF_90d"] = pred_rf
        result_df["Pred_XGB_90d"] = pred_xgb
        csv = result_df.to_csv(index=False)
        st.download_button("Download prediction row as CSV", data=csv, file_name=f"{selected}_prediction_row.csv", mime="text/csv")

# Optional: show model metrics on demand if models available and dataset present
if st.checkbox("Show model test metrics (quick)", value=False):
    # quick local eval using last available train/test split approach
    # (this does not retrain; it's a simple illustrative convenience)
    try:
        y = df["Target_90d"]
        X_all = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(method='ffill').fillna(method='bfill')
        # small split
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X_all, y, test_size=0.2, random_state=42)
        ypred_rf = rf_model.predict(Xte)
        ypred_xgb = xgb_model.predict(Xte)
        rf_rmse = np.sqrt(((yte-ypred_rf)**2).mean())
        xgb_rmse = np.sqrt(((yte-ypred_xgb)**2).mean())
        st.write(f"RF RMSE (quick): {rf_rmse:.4f}")
        st.write(f"XGB RMSE (quick): {xgb_rmse:.4f}")
    except Exception as e:
        st.write("Could not compute quick metrics:", e)

st.markdown("---")
st.write("App built for CAIP project — make sure to place preprocessed CSVs and model .pkl files in the repo before deploying.")
