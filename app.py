import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import os

# ======================
# LOAD MODEL
# ======================
model = load_model("cnn_gender_model.keras")

SR = 16000
MAX_LEN = 130

# ======================
# FEATURE EXTRACTION
# ======================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    # pad / truncate
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :MAX_LEN]

    # normalize
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)

    return features.astype(np.float32)

# ======================
# PREDICTION FUNCTION
# ======================
def predict_gender(file_path):

    audio = AudioSegment.from_wav(file_path)

    probs = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        if chunk.dBFS == float("-inf") or chunk.dBFS < -55:
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = model.predict(feat, verbose=0)[0][0]
        probs.append(prob)

    if len(probs) == 0:
        return "No speech detected", 0.0

    avg_prob = np.mean(probs)

    # ======================
    # FINAL DECISION RULE
    # ======================
    if avg_prob >= 0.5:
        label = "MALE"
    else:
        label = "FEMALE"

    return label, avg_prob


# ======================
# STREAMLIT UI
# ======================
st.title("🎤 Assamese Gender Prediction App")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:

    path = "uploaded.wav"

    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(path)

    label, prob = predict_gender(path)

    st.success(f"Prediction: {label}")

    st.write("Male probability:", float(prob))
