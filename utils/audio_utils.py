# utils/audio_utils.py
import numpy as np
import librosa

def extract_mfcc_from_file(filepath, sr=22050, n_mfcc=13, duration=None):
    """
    Load audio file lalu ekstrak MFCC.
    Returns feature vector (1D) yang siap dipakai model.
    """
    try:
        y, sr = librosa.load(filepath, sr=sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # ringkas: ambil mean dan std tiap koef
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std  = np.std(mfcc, axis=1)
        features = np.concatenate([mfcc_mean, mfcc_std])
        return features
    except Exception as e:
        print("Error extract_mfcc:", e)
        return None
