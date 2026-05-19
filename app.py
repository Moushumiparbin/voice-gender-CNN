import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile
from utils import predict_audio

st.set_page_config(page_title="Speech Gender Classification", layout="centered")

st.title("🎤 Assamese Speech Gender Classification")
st.write("Upload a WAV file to predict gender")

import keras
import streamlit as st

@st.cache_resource
def load_model():
    model = keras.models.load_model(
        "hybrid_model.keras",
        compile=False,
        custom_objects={}
    )
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.audio(uploaded_file)

    st.info("Processing...")

    label, confidence = predict_audio(file_path, model)

    st.success(f"🎯 Prediction: {label}")
    st.write(f"📊 Confidence: {confidence*100:.2f}%")
