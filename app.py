# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from food_module import predict_food_impact
from health_suggestions import suggest

st.set_page_config(page_title="Advanced Diabetes Prediction System", layout="wide")
st.title("created by MEERAN UBAITHULLAH")
# -------------------------
# 1️⃣ Imports
# -------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib  # ✅ Use joblib, not pickle


import streamlit as st

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    }

    /* Title Styling */
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #0A2647;
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }

    /* Header Styling */
    .section-header {
        font-size: 26px;
        font-weight: 600;
        color: white;
        background: linear-gradient(90deg, #2E8BC0, #145DA0);
        padding: 10px 20px;
        border-radius: 8px;
        margin-top: 20px;
    }

/* Section Container Card */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Green AI Section Header */
.ai-header {
    font-size: 24px;
    font-weight: 600;
    color: white;
    background: linear-gradient(90deg, #11998e, #38ef7d);
    padding: 12px 20px;
    border-radius: 8px;
    margin-top: 20px;
}

/* Button Styling */
div.stButton > button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}

div.stButton > button:hover {
    opacity: 0.85;
}

/* Suggestion Box */
.suggestion-box {
    background-color: #f0f9ff;
    padding: 15px;
    border-left: 6px solid #0284c7;
    border-radius: 6px;
    margin-top: 15px;
}

/* Chart container */
.chart-box {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-top: 15px;
}
    </style>
""", unsafe_allow_html=True)

# -------------------------
# 2️⃣ Load models and scalers
# -------------------------
lstm_model = load_model("multivariate_lstm.h5", compile=False)
lstm_scaler = joblib.load("multivariate_lstm_scaler.pkl")

# -------------------------
# 3️⃣ Streamlit app title
# -------------------------
st.markdown('<div class="main-title">🩺 Advanced Diabetes Prediction System</div>', unsafe_allow_html=True)
# -------------------------
# 1️⃣ Current Risk Prediction
# -------------------------
st.markdown('<div class="section-header">Current Risk Prediction</div>', unsafe_allow_html=True)

# Load classifier model
clf_model = pickle.load(open("classification_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict Current Risk"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    result = clf_model.predict(input_scaled)
    if result[0] == 1:
        st.error("⚠️ High Diabetes Risk")
    else:
        st.success("✅ Low Diabetes Risk")

# -------------------------
# 2️⃣ 30-Day Multivariate Glucose Prediction
# -------------------------
st.markdown('<div class="ai-header">🔮 Future 30 Days Glucose Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

if st.button("Show Future Trend"):
    # Load multivariate LSTM model and scaler
    lstm_model = load_model("multivariate_lstm.h5", compile=False)
    import joblib
    lstm_scaler = joblib.load("multivariate_lstm_scaler.pkl")

    # Load last 5 rows from multivariate CSV
    df = pd.read_csv("multivariate_glucose_timeseries.csv")
    if len(df) < 5:
        st.error("❌ CSV must have at least 5 rows of data")
    else:
        last_seq = df.drop(columns=["Day"]).values[-5:]  # last 5 rows
        scaled_seq = lstm_scaler.transform(last_seq)
        sequence = scaled_seq.reshape(1, 5, scaled_seq.shape[1])  # (1, timesteps, features)

        # Predict next 30 days
        future = []
        for _ in range(30):
            pred = lstm_model.predict(sequence, verbose=0)
            future.append(pred[0][0])
            # shift sequence and insert new glucose prediction
            sequence = np.roll(sequence, -1, axis=1)
            sequence[0, -1, 0] = pred[0][0]

            st.markdown('</div>', unsafe_allow_html=True)

        # Display chart
        st.markdown('<div class="chart-box"><h3>📈 30-Day Glucose Forecast</h3>', unsafe_allow_html=True)
        st.line_chart(future)
        st.markdown('</div>', unsafe_allow_html=True)
# -------------------------
# 3️⃣ Food Impact & Health Suggestions
# -------------------------
st.markdown('<div class="ai-header">🥗 Food Impact & AI Health Suggestions</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

food = st.text_input("Enter a food item:", "White Rice")
steps = st.number_input("Steps today", value=3000)
sleep = st.number_input("Sleep hours", value=6.5)
stress = st.number_input("Stress level 1-10", value=5)

if st.button("Get Food & Health Advice"):
    # Food impact
    impact = predict_food_impact(food)
    st.write(f"Glucose impact of **{food}**: {impact}")

    # AI health suggestions
    tips = suggest(glucose=glucose,
                   steps=steps,
                   sleep=sleep,
                   stress=stress)
    st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
    st.markdown("💡 **AI Suggestions:**")
    for tip in tips:
        st.write("- " + tip)
        st.markdown('</div>', unsafe_allow_html=True)