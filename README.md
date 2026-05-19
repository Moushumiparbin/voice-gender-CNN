# 🎤 Assamese Speech Gender Classification using Deep Learning

This project is a **Speech Processing + Deep Learning system** that classifies the gender (Male/Female) of a speaker from Assamese speech audio using MFCC-based feature extraction and a Hybrid CNN-LSTM model.

The system is deployed using **Streamlit Community Cloud** for real-time inference.

---

## 🚀 Live Demo
If deployed:
👉 https://your-streamlit-app-link.streamlit.app/

---

## 📌 Project Overview

The goal of this project is to build an **automatic gender classification system** from speech signals.

We compare three deep learning architectures:
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Hybrid CNN + LSTM (Best Model)

---

## 📊 Dataset

- Language: Assamese
- Audio format: `.wav`
- Speaker-based dataset
- Each speaker labeled with:
  - Gender (Male/Female)
  - Age

---

## 🎯 Feature Extraction

We extract **39-dimensional MFCC features**:

- 13 MFCC coefficients
- 13 Delta MFCC
- 13 Delta-Delta MFCC

Each audio is segmented into **3-second chunks**, and features are padded/truncated to fixed length.

Final input shape:
