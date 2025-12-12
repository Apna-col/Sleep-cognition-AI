import streamlit as st
import pandas as pd
import joblib
import numpy as np
from streamlit_option_menu import option_menu

# -----------------------------------------------------------
# Load model and encoders
# -----------------------------------------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoder.pkl")  # contains {'mem': le_mem, 'react': le_react}

le_mem = encoders['mem']
le_react = encoders['react']

# -----------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------
st.set_page_config(page_title="Cognitive Performance Predictor",
                   page_icon="🧠",
                   layout="wide")

# -----------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        "AI Dashboard Menu",
        ["Home", "Predict Performance", "About"],
        icons=["house", "activity", "info-circle"],
        menu_icon="cast",
        default_index=0
    )

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if selected == "Home":
    st.title("🧠 Cognitive Performance Prediction Dashboard")
    st.subheader("AI model trained on sleep behavior data")
    st.info("This AI dashboard is made for School Project purpose.Not recommended for medical and research grade purpose.")
    st.markdown("""
    This smart dashboard predicts:

    - **Memory Accuracy** → Good / Fine / Poor  
    - **Reaction Speed** → Fast / Medium / Slow  

    Based on the following sleep habits:

    ✓ Sleep Duration  
    ✓ Sleep Efficiency  
    ✓ Room Temperature  
    ✓ Sound Level  
    ✓ Light Intensity  
    ✓ Awake Time  
    ✓ Bed Time  
    """)
   

# -----------------------------------------------------------
# PREDICTION PAGE
# -----------------------------------------------------------
if selected == "Predict Performance":
    st.header("📊 Predict Cognitive Performance")

    col1, col2 = st.columns(2)

    with col1:
        sleep_dur = st.number_input("Sleep Duration (hours)", 0.0, 12.0, 7.0)
        sleep_effi = st.number_input("Sleep Efficiency (%)", 0.0, 100.0, 85.0)
        awake_time = st.number_input("Awake Time (minutes)", 0.0, 120.0, 5.0)

    with col2:
        room_temp = st.selectbox("Room Temperature", ["Low", "Normal", "High"])
        sound = st.selectbox("Sound Level", ["None", "Slow", "Loud"])
        light = st.selectbox("Light Intensity", ["Dark", "Dim", "Bright"])

    bed_time = st.time_input("Bed Time")

    # Convert bed time into hour + minute
    bed_hour = bed_time.hour
    bed_minute = bed_time.minute

    # Create DataFrame for prediction
    input_data = pd.DataFrame([{
        "Sleep_Dur": sleep_dur,
        "Sleep_Effi": sleep_effi,
        "Room_*C": room_temp,
        "Sound": sound,
        "Light": light,
        "Awake_tim": awake_time,
        "Bed_hour": bed_hour,
        "Bed_minute": bed_minute
    }])

    if st.button("Predict"):
        # Predict using model
        pred = model.predict(input_data)[0]

        # Decode labels
        mem_pred = le_mem.inverse_transform([pred[0]])[0]
        react_pred = le_react.inverse_transform([pred[1]])[0]

        st.success("Prediction Completed Successfully!")
        c1, c2 = st.columns(2)

        with c1:
            st.metric("🧠 Memory Accuracy", mem_pred)

        with c2:
            st.metric("⚡ Reaction Speed", react_pred)

# -----------------------------------------------------------
# ABOUT PAGE
# -----------------------------------------------------------
if selected == "About":
    st.title("ℹ️ About This Project")
    st.write("""
    This dashboard uses a **multi-output machine learning model**  
    trained on real sleep behavior data.

    The AI predicts:

    - **Memory Accuracy**
    - **Reaction Speed**

    Using sleep factors like duration, efficiency, temp, sound,
    light, awake time, and bed time.
    """)




