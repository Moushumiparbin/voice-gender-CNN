import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa
from collections import Counter

SR = 16000
MAX_LEN = 130
EPS = 1e-8

MODEL_PATH = "cnn_gender_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# =========================
# EXACT FEATURE EXTRACTION
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.vstack([mfcc, delta, delta2])

    feat = feat[:, :MAX_LEN] if feat.shape[1] > MAX_LEN else np.pad(
        feat, ((0,0),(0, MAX_LEN - feat.shape[1]))
    )

    feat = (feat - np.mean(feat, axis=1, keepdims=True)) / (
        np.std(feat, axis=1, keepdims=True) + EPS
    )

    return feat.astype(np.float32)

# =========================
# CRITICAL FIX: SAME AUDIO NORMALIZATION AS TRAINING
# =========================
def preprocess_audio(audio):

    # remove silence base
    if audio.dBFS == float("-inf"):
        return None

    # 🔥 SAME AS COLAB PIPELINE IDEA
    audio = audio.apply_gain(-audio.dBFS)

    return audio


# =========================
# PREDICT FUNCTION
# =========================
def predict(file_path):

    audio = AudioSegment.from_wav(file_path)
    audio = preprocess_audio(audio)

    if audio is None:
        return None, 0

    probs = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        if chunk.dBFS < -55:
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = model.predict(feat, verbose=0)[0][0]

        st.write("Chunk prob:", float(prob))  # DEBUG

        probs.append(prob)

    if len(probs) == 0:
        return None, 0

    avg = float(np.mean(probs))

    st.write("Average prob:", avg)

    label = "MALE" if avg > 0.5 else "FEMALE"
    confidence = max(avg, 1 - avg)

    return label, confidence


# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Voice Gender Classification")

file = st.file_uploader("Upload WAV", type=["wav"])

if file:
    with open("temp.wav", "wb") as f:
        f.write(file.read())

    st.audio(file)

    if st.button("Predict"):
        label, conf = predict("temp.wav")
        st.success(label)
        st.info(conf)
