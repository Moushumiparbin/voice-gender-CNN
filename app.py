import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from pydub import AudioSegment

# =========================
# CONFIG
# =========================
SR = 16000
MAX_LEN = 130
EPS = 1e-8
MODEL_PATH = "cnn_gender_model.keras"

AudioSegment.converter = "ffmpeg"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# =========================
# FEATURE EXTRACTION (SAME AS COLAB)
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])  # (39, T)

    # pad / truncate
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :MAX_LEN]

    # normalize
    features = (features - np.mean(features, axis=1, keepdims=True)) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )

    return features.astype(np.float32)

# =========================
# CLEAN PREDICTION (FIXED VERSION)
# =========================
def predict_gender(file_path):

    audio = AudioSegment.from_wav(file_path)

    probs = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        # skip silence
        if chunk.dBFS == float("-inf") or chunk.dBFS < -55:
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = float(model.predict(feat, verbose=0)[0][0])

        probs.append(prob)

        # 🔍 DEBUG VIEW (for viva)
        st.write("Chunk prob:", prob)

    if len(probs) == 0:
        return None, 0, 0

    # =========================
    # STABLE AGGREGATION
    # =========================
    avg_prob = np.mean(probs)

    # FINAL DECISION (SMOOTHED THRESHOLD)
    if avg_prob > 0.55:
        label = "MALE"
    elif avg_prob < 0.45:
        label = "FEMALE"
    else:
        label = "UNCERTAIN"

    confidence = abs(avg_prob - 0.5) * 2  # normalized confidence

    return label, confidence, avg_prob

# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Voice Gender Classification (CNN)")
st.write("Upload a WAV file for prediction")

uploaded_file = st.file_uploader("Upload WAV", type=["wav"])

if uploaded_file is not None:

    with open("temp_input.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        label, conf, avg_prob = predict_gender("temp_input.wav")

        if label is None:
            st.warning("No speech detected in audio")
        else:
            st.success(f"Prediction: {label}")

            st.info(f"Avg Male Probability: {avg_prob:.3f}")

            st.info(f"Confidence (normalized): {conf:.3f}")

            st.write("Raw decision logic:")
            st.write("MALE if prob > 0.55 else FEMALE")
