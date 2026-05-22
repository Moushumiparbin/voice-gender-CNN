import streamlit as st
import numpy as np
import os
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from utils import extract_features
from collections import Counter

# =========================
# FFmpeg setup (IMPORTANT)
# =========================
AudioSegment.converter = "ffmpeg"

# =========================
# LOAD MODEL (FIXED)
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_gender_model.keras")
model = load_model(MODEL_PATH, compile=False)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Voice Gender Detection", layout="centered")

st.title("🎤 Voice Gender Classification (CNN)")
st.write("Upload a WAV file to predict gender")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

# =========================
# PREDICTION FUNCTION
# =========================
def predict(file_path, threshold=0.5):

    audio = AudioSegment.from_wav(file_path)
    predictions = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        # skip silence
        if chunk.dBFS == float("-inf") or chunk.dBFS < -55:
            continue

        temp_file = "temp.wav"
        chunk.export(temp_file, format="wav")

        # feature extraction
        feat = extract_features(temp_file)

        # CNN input shape (1, 39, 130, 1)
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = model.predict(feat, verbose=0)[0][0]

        label = "MALE" if prob > threshold else "FEMALE"
        predictions.append(label)

    if len(predictions) == 0:
        return None, 0

    final_label = Counter(predictions).most_common(1)[0][0]
    confidence = Counter(predictions)[final_label] / len(predictions)

    return final_label, confidence

# =========================
# UI LOGIC
# =========================
if uploaded_file is not None:

    file_path = "temp_uploaded.wav"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        label, conf = predict(file_path)

        if label is None:
            st.warning("⚠️ No speech detected")
        else:
            st.success(f"🎯 Prediction: {label}")
            st.info(f"📊 Confidence: {conf:.3f}")
