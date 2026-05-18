import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load model only once
model = load_model("hybrid_model.h5")

st.title("🎤 Voice Gender Classification (CNN + LSTM Hybrid)")
st.write("Upload a WAV file and get prediction instantly")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

# Feature extraction
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# Chunk processing (same logic as your project)
def process_audio(file):
    y, sr = librosa.load(file, sr=16000)

    chunk_size = 3 * sr
    predictions = []

    for i in range(0, len(y), chunk_size):
        chunk = y[i:i+chunk_size]

        if len(chunk) < sr:
            continue

        feat = extract_features(chunk, sr)
        feat = feat.reshape(1, -1)

        pred = model.predict(feat, verbose=0)[0]
        predictions.append(pred)

    return np.array(predictions)

if uploaded_file is not None:
    st.audio(uploaded_file)

    preds = process_audio(uploaded_file)

    final_pred = np.mean(preds, axis=0)
    label = np.argmax(final_pred)

    st.subheader("Prediction Result")

    if label == 0:
        st.success("🎯 Male")
    else:
        st.success("🎯 Female")

    st.write("Confidence: ", float(np.max(final_pred)) * 100, "%")