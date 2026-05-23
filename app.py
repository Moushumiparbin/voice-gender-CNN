import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from pydub import AudioSegment

AudioSegment.converter = "ffmpeg"

SR = 16000
MAX_LEN = 130
EPS = 1e-8

MODEL_PATH = "cnn_gender_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :MAX_LEN]
    features = (features - np.mean(features, axis=1, keepdims=True)) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )
    return features[np.newaxis, ..., np.newaxis].astype(np.float32)

def predict(file_path):
    feat = extract_features(file_path)
    prob = float(model.predict(feat, verbose=0)[0][0])
    st.write("Model prob:", prob)  # DEBUG
    label = "MALE" if prob > 0.5 else "FEMALE"
    confidence = max(prob, 1 - prob)
    return label, confidence

st.title("🎤 Voice Gender Classification (CNN)")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    with open("temp_uploaded.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.audio(uploaded_file)
    if st.button("Predict Gender"):
        label, conf = predict("temp_uploaded.wav")
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf:.3f}")
