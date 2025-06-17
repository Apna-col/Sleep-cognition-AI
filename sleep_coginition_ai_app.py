import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import hashlib
import json
from datetime import datetime
from streamlit_option_menu import option_menu

# --- Page Setup ---
st.set_page_config(page_title="Sleep & Cognition AI", page_icon="brain_sleep_icon.png", layout="wide")

# --- Sidebar Navigation ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Model Evaluation", "Predict"],
        icons=["house", "graph-up", "activity"],
        menu_icon="cast",
        default_index=0
    )

# --- File Upload ---
st.sidebar.subheader("📁 Upload Your Sleep-Cognition Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded and loaded successfully!")
else:
    file_path = "my_sleep_data.csv"
    if not os.path.exists(file_path):
        st.error(f"Data file '{file_path}' not found. Please upload one.")
        st.stop()
    data = pd.read_csv(file_path)
    st.sidebar.info("Using default data: my_sleep_data.csv")

# --- Required Columns ---
required_cols = [
    'sleep_duration', 'sleep_efficiency', 'rem_percentage',
    'deep_sleep_percentage', 'wake_after_sleep',
    'reaction_time_ms', 'memory_accuracy'
]
if not all(col in data.columns for col in required_cols):
    st.error("CSV file is missing required columns. Required: " + ", ".join(required_cols))
    st.stop()

# --- Data Preparation ---
sleep_data = data[[
    'sleep_duration', 'sleep_efficiency', 'rem_percentage',
    'deep_sleep_percentage', 'wake_after_sleep'
]]
cognitive_data = data[['reaction_time_ms', 'memory_accuracy']]

X = sleep_data
y_rt = cognitive_data['reaction_time_ms']
y_mem = cognitive_data['memory_accuracy']

# --- Hashing Utilities ---
def hash_dataframe(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def save_hash_to_file(hash_str, filename="last_data_hash.json"):
    with open(filename, "w") as f:
        json.dump({"hash": hash_str}, f)

def load_hash_from_file(filename="last_data_hash.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f).get("hash", None)
    return None

# --- Training Log Utility ---
def log_training_metrics(mae_rt, r2_rt, mae_mem, r2_mem, log_file="model_training_log.csv"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{
        "timestamp": now,
        "mae_reaction_time": mae_rt,
        "r2_reaction_time": r2_rt,
        "mae_memory_accuracy": mae_mem,
        "r2_memory_accuracy": r2_mem
    }])
    if os.path.exists(log_file):
        old_log = pd.read_csv(log_file)
        full_log = pd.concat([old_log, new_entry], ignore_index=True)
    else:
        full_log = new_entry
    full_log.to_csv(log_file, index=False)

# --- Train Only If Data Changed ---
current_hash = hash_dataframe(data)
saved_hash = load_hash_from_file()

if current_hash != saved_hash:
    st.info("🔁 New data detected. Retraining model...")

    model_rt = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rt.fit(X, y_rt)
    joblib.dump(model_rt, "model_rt.pkl")

    model_mem = RandomForestRegressor(n_estimators=100, random_state=42)
    model_mem.fit(X, y_mem)
    joblib.dump(model_mem, "model_mem.pkl")

    save_hash_to_file(current_hash)
else:
    model_rt = joblib.load("model_rt.pkl")
    model_mem = joblib.load("model_mem.pkl")

# --- MAIN SECTIONS ---
if selected == "Home":
    st.title("🧠 Sleep & Cognition AI Dashboard")
    st.markdown("Upload your sleep and cognition data to train an AI model and predict performance based on your sleep pattern.")

elif selected == "Model Evaluation":
    st.header("📊 Model Evaluation")
    pred_rt_all = model_rt.predict(X)
    pred_mem_all = model_mem.predict(X)

    mae_rt = mean_absolute_error(y_rt, pred_rt_all)
    r2_rt = r2_score(y_rt, pred_rt_all)
    mae_mem = mean_absolute_error(y_mem, pred_mem_all)
    r2_mem = r2_score(y_mem, pred_mem_all)

    if current_hash != saved_hash:
        log_training_metrics(mae_rt, r2_rt, mae_mem, r2_mem)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE (Reaction Time)", f"{mae_rt:.2f} ms")
        st.metric("R² (Reaction Time)", f"{r2_rt:.3f}")
    with col2:
        st.metric("MAE (Memory Accuracy)", f"{mae_mem:.3f}")
        st.metric("R² (Memory Accuracy)", f"{r2_mem:.3f}")

    st.subheader("📈 Learning Progress")
    if 'daily_mae_rt' not in st.session_state:
        st.session_state['daily_mae_rt'] = []
        st.session_state['daily_r2_rt'] = []
        st.session_state['daily_mae_mem'] = []
        st.session_state['daily_r2_mem'] = []

    st.session_state['daily_mae_rt'].append(mae_rt)
    st.session_state['daily_r2_rt'].append(r2_rt)
    st.session_state['daily_mae_mem'].append(mae_mem)
    st.session_state['daily_r2_mem'].append(r2_mem)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("📈 Daily AI Learning Progress")
    axs[0, 0].plot(st.session_state['daily_mae_rt'], label='MAE - RT', color='blue')
    axs[0, 1].plot(st.session_state['daily_r2_rt'], label='R² - RT', color='green')
    axs[1, 0].plot(st.session_state['daily_mae_mem'], label='MAE - Mem', color='red')
    axs[1, 1].plot(st.session_state['daily_r2_mem'], label='R² - Mem', color='purple')
    for ax in axs.flat:
        ax.set_xlabel("Days")
        ax.set_ylabel("Score")
        ax.legend()
    st.pyplot(fig)

    st.subheader("📜 Model Training History")
    log_file_path = "model_training_log.csv"
    if os.path.exists(log_file_path):
        log_df = pd.read_csv(log_file_path)
        st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True)
    else:
        st.info("ℹ️ No training history available yet.")

elif selected == "Predict":
    st.header("📌 Sleep Inputs")
    sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, step=0.1)
    sleep_efficiency = st.slider("Sleep Efficiency", 0.0, 1.0, 0.85, step=0.01)
    rem_percentage = st.slider("REM Sleep (%)", 0.0, 0.5, 0.2, step=0.01)
    deep_sleep_percentage = st.slider("Deep Sleep (%)", 0.0, 0.5, 0.25, step=0.01)
    wake_after_sleep = st.slider("Wake After Sleep Onset (min)", 0, 120, 30, step=1)

    if st.button("🔍 Predict Cognitive Performance"):
        input_df = pd.DataFrame([{
            'sleep_duration': sleep_duration,
            'sleep_efficiency': sleep_efficiency,
            'rem_percentage': rem_percentage,
            'deep_sleep_percentage': deep_sleep_percentage,
            'wake_after_sleep': wake_after_sleep
        }])

        pred_rt = model_rt.predict(input_df)[0]
        pred_mem = model_mem.predict(input_df)[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Reaction Time (ms)", f"{pred_rt:.2f}")
        with col2:
            st.metric("Memory Accuracy", f"{pred_mem:.3f}")

        fig2, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(["Reaction Time", "Memory Accuracy"], [pred_rt, pred_mem], color=["#4f8df7", "#34c759"])
        ax.set_ylim([0, max(pred_rt + 50, 1)])
        ax.set_ylabel("Predicted Value")
        ax.set_title("🧠 Cognitive Prediction Overview")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 5, f"{height:.2f}", ha='center', fontsize=10)
        st.pyplot(fig2)
