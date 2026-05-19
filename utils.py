import numpy as np
import librosa
from collections import Counter
import uuid

SR = 16000
N_MFCC = 13
MAX_LEN = 130
CHUNK_DURATION = 3  # seconds


# ================= FEATURE EXTRACTION =================
def extract_features(y, sr):

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    if features.shape[1] < MAX_LEN:
        features = np.pad(
            features,
            ((0, 0), (0, MAX_LEN - features.shape[1])),
            mode='constant'
        )
    else:
        features = features[:, :MAX_LEN]

    return features


# ================= AUDIO CHUNKING (NO PYDUB) =================
def chunk_audio(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    chunk_size = SR * CHUNK_DURATION

    chunks = []

    for i in range(0, len(y), chunk_size):

        chunk = y[i:i + chunk_size]

        if len(chunk) < SR:  # ignore too small chunks
            continue

        if np.max(np.abs(chunk)) < 0.01:  # silence filter
            continue

        chunks.append((chunk, sr))

    return chunks


# ================= PREDICTION =================
def predict_audio(file_path, model):

    chunks = chunk_audio(file_path)
    results = []

    for chunk, sr in chunks:

        features = extract_features(chunk, sr)

        # CNN-LSTM format
        features = np.transpose(features, (1, 0))
        features = np.expand_dims(features, axis=0)

        pred = model.predict(features, verbose=0)

        label = "Female" if pred[0][0] > 0.5 else "Male"
        results.append(label)

    if len(results) == 0:
        return "No valid audio detected", 0.0

    counts = Counter(results)
    final = counts.most_common(1)[0][0]
    confidence = counts[final] / len(results)

    return final, confidence
