import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("cnn_gender_model.keras", compile=False)

# =========================
# CONSTANTS
# =========================
SR = 16000
MAX_LEN = 130
EPS = 1e-8

# =========================
# FEATURE EXTRACTION (SAME AS COLAB)
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    # normalize audio (IMPORTANT FIX)
    y = librosa.util.normalize(y)

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
# PREDICTION (STABLE VERSION)
# =========================
def predict_gender(file_path):

    feat = extract_features(file_path)
    feat = feat[np.newaxis, ..., np.newaxis]

    prob = float(model.predict(feat, verbose=0)[0][0])

    # FINAL DECISION RULE (VERY IMPORTANT)
    if prob > 0.55:
        label = "MALE"
    elif prob < 0.45:
        label = "FEMALE"
    else:
        label = "UNCERTAIN"

    return label, prob

# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Voice Gender Classification (CNN)")
st.write("Upload a WAV file and get prediction")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:

    file_path = "temp.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        label, prob = predict_gender(file_path)

        st.success(f"Prediction: {label}")
        st.info(f"Male Probability: {prob:.3f}")

        st.write("Decision Rule:")
        st.code("Male if prob > 0.55 else Female else Uncertain")
