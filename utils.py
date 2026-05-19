import numpy as np
import librosa
from collections import Counter

SR = 16000
N_MFCC = 13
MAX_LEN = 130
CHUNK_DURATION = 3  # seconds


# ================= FEATURE EXTRACTION (MATCH COLAB) =================
def extract_features(y):

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])  # (39, time)

    # FIX LENGTH = 130
    if features.shape[1] < MAX_LEN:
        features = np.pad(
            features,
            ((0, 0), (0, MAX_LEN - features.shape[1])),
            mode='constant'
        )
    else:
        features = features[:, :MAX_LEN]

    return features


# ================= AUDIO CHUNKING (MATCH COLAB STYLE) =================
def chunk_audio(file_path):

    y, sr = librosa.load(file_path, sr=SR)
    chunk_size = SR * CHUNK_DURATION

    chunks = []

    for i in range(0, len(y), chunk_size):
        chunk = y[i:i + chunk_size]

        # same logic as your Colab (silence filter)
        if len(chunk) < SR:
            continue

        if np.max(np.abs(chunk)) < 0.01:
            continue

        chunks.append(chunk)

    return chunks


# ================= PREDICTION =================
def predict_audio(file_path, model):

    chunks = chunk_audio(file_path)
    results = []

    for chunk in chunks:

        features = extract_features(chunk)

        # (39,130) → (130,39)
        features = np.transpose(features, (1, 0))
        features = features[np.newaxis, ...]

        pred = model.predict(features, verbose=0)

        label = "Female" if pred[0][0] > 0.5 else "Male"
        results.append(label)

    if len(results) == 0:
        return "No valid audio detected", 0.0

    counts = Counter(results)
    final = counts.most_common(1)[0][0]
    confidence = counts[final] / len(results)

    return final, confidence
