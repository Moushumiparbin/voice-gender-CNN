import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa

# =========================
# SETTINGS (MUST MATCH TRAINING)
# =========================
SR = 16000
MAX_LEN = 130
EPS = 1e-8

MODEL_PATH = "cnn_gender_model.keras"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_safe():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model_safe()

# =========================
# FEATURE EXTRACTION (EXACT COPY FROM COLAB)
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])  # (39, T)

    # FIX TIME LENGTH (130)
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :MAX_LEN]

    # NORMALIZATION (EXACT TRAIN MATCH)
    features = (features - np.mean(features, axis=1, keepdims=True)) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )

    return features.astype(np.float32)

# =========================
# PREDICTION (NO CHUNKING — FIXES 0.5 PROBLEM)
# =========================
def predict(file_path):

    feat = extract_features(file_path)

    feat = feat[np.newaxis, ..., np.newaxis]  # (1,39,130,1)

    prob = float(model.predict(feat, verbose=0)[0][0])

    # FINAL DECISION (CALIBRATED BUT SIMPLE)
    if prob >= 0.52:
        label = "MALE"
    elif prob <= 0.48:
        label = "FEMALE"
    else:
        label = "UNCERTAIN"

    confidence = max(prob, 1 - prob)

    return label, confidence

# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Voice Gender Classification (Fixed 39 MFCC Model)")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    file_path = "temp.wav"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        label, conf = predict(file_path)

        if label is None:
            st.warning("No valid audio detected")
        else:
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf:.3f}")
