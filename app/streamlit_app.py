import streamlit as st
import json
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import pandas as pd

# -------------------------
# Config
# -------------------------
TRAITS = ["EXT", "NEU", "AGR", "CON", "OPN"]
MODEL_DIR = "../LightGBModel/lgbm_20q_scaled"   # ✅ matches your structure

# -------------------------
# Load Models + Scalers
# -------------------------
models, scalers = {}, {}
for trait in TRAITS:
    with open(os.path.join(MODEL_DIR, f"lgbm_{trait}.pkl"), "rb") as f:
        models[trait] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f"scaler_{trait}.pkl"), "rb") as f:
        scalers[trait] = pickle.load(f)

# -------------------------
# Load Questions
# -------------------------
with open("questions.json", "r") as f:
    questions = json.load(f)

# Flatten question list into dict {trait: [questions]}
question_map = {t: [] for t in TRAITS}
for q in questions:
    question_map[q["trait"]].append(q)

# -------------------------
# Helper: Bin classification
# -------------------------
def get_bin(score):
    """Assign Low / Medium / High based on fixed cutoffs"""
    if score <= 33:
        return "Low"
    elif score <= 66:
        return "Medium"
    else:
        return "High"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Big Five Personality Predictor", layout="wide")
st.title("🧠 Big Five Personality Prediction (20 Questions)")
st.write("Answer 20 short questions to estimate your personality traits.")

with st.form("personality_form"):
    responses = {}
    for trait in TRAITS:
        st.subheader(trait)
        for q in question_map[trait]:
            responses[q["id"]] = st.slider(
                q["text"], min_value=1, max_value=5, value=3, step=1
            )
    submitted = st.form_submit_button("🔮 Predict Personality")

# -------------------------
# Prediction
# -------------------------
if submitted:
    # Map user responses to training feature names
    input_features = [
        responses["EXT1"], responses["EXT2"], responses["EXT3"], responses["EXT4"],
        responses["AGR1"], responses["AGR2"], responses["AGR3"], responses["AGR4"],
        responses["CON1"], responses["CON2"], responses["CON3"], responses["CON4"],
        responses["NEU1"], responses["NEU2"], responses["NEU3"], responses["NEU4"],
        responses["OPN1"], responses["OPN2"], responses["OPN3"], responses["OPN4"]
    ]

    # Rename features to match training
    feature_names = [
        "EXT1","EXT2","EXT3","EXT4",
        "AGR1","AGR2","AGR3","AGR4",
        "CON1","CON2","CON3","CON4",
        "NEU1","NEU2","NEU3","NEU4",
        "OPN1","OPN2","OPN3","OPN4"
    ]

    X_input = pd.DataFrame([input_features], columns=feature_names)

    predictions = {}
    bins = {}

    for trait in TRAITS:
        model = models[trait]
        scaler = scalers[trait]

        # Predict in scaled space, then inverse transform
        scaled_pred = model.predict(X_input)
        final_pred = scaler.inverse_transform(
            scaled_pred.reshape(-1, 1)
        ).ravel()[0]

        # Normalize to 0–100 for display
        norm_score = np.clip((final_pred + 3) * 20, 0, 100)
        predictions[trait] = norm_score
        bins[trait] = get_bin(norm_score)

    # -------------------------
    # Display Results
    # -------------------------
    st.subheader("📊 Your Personality Profile")

    # Radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[predictions[t] for t in TRAITS],
        theta=TRAITS,
        fill='toself',
        name="Your Traits"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table summary
    st.subheader("📝 Detailed Results")
    for trait in TRAITS:
        st.write(f"**{trait}**: {predictions[trait]:.1f} → {bins[trait]}")

    st.success("✅ Prediction complete!")
