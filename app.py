import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("cnn_gender_model.keras", compile=False)

# =========================
# CONSTANTS (same as training)
# =========================
SR = 16000
MAX_LEN = 130
EPS = 1e-8

# =========================
# FEATURE EXTRACTION (FIXED - MATCH TRAINING)
# =========================
def extract_features(file_path):

    # IMPORTANT: use ONLY librosa (no pydub)
    y, sr = librosa.load(file_path, sr=SR)

    # normalize waveform
    y = y / (np.max(np.abs(y)) + 1e-8)

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # combine (39, T)
    feat = np.vstack([mfcc, delta, delta2])

    # FIX LENGTH (130 frames)
    if feat.shape[1] < MAX_LEN:
        feat = np.pad(feat, ((0,0),(0, MAX_LEN - feat.shape[1])))
    else:
        feat = feat[:, :MAX_LEN]

    # IMPORTANT NORMALIZATION (CRITICAL FIX)
    feat = (feat - np.mean(feat)) / (np.std(feat) + EPS)

    return feat.astype(np.float32)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_gender(file_path):

    feat = extract_features(file_path)

    # reshape for CNN
    feat = feat[np.newaxis, ..., np.newaxis]

    prob = float(model.predict(feat, verbose=0)[0][0])

    # stable decision logic
    if prob > 0.60:
        label = "MALE"
    elif prob < 0.40:
        label = "FEMALE"
    else:
        label = "UNCERTAIN"

    return label, prob

# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Gender Classification from Voice (CNN Model)")
st.write("Upload a WAV file for prediction")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:

    file_path = "temp.wav"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        label, prob = predict_gender(file_path)

        st.success(f"Prediction: {label}")
        st.info(f"Male Probability: {prob:.4f}")

        st.write("Decision Rule:")
        st.code("MALE if prob > 0.60 else FEMALE if prob < 0.40 else UNCERTAIN")

        st.write("Debug Info (important for viva):")
        st.write("Feature shape:", extract_features(file_path).shape)
        st.write("Mean:", np.mean(extract_features(file_path)))
        st.write("Std:", np.std(extract_features(file_path)))
