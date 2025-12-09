import numpy as np
import joblib
from ml.yamnet_feature_extractor import YamnetFeatureExtractor
from utils.audio_loader import load_audio_16k

class QuranDetector:
    def __init__(self):
        self.classifier = joblib.load("model/logistic_regression_model.pkl")
        self.yamnet = YamnetFeatureExtractor()

    def predict(self, audio_path):
        # Load audio & resample ke 16k
        waveform = load_audio_16k(audio_path)

        # Extract YAMNet embedding
        embedding = self.yamnet.extract_embedding(waveform)

        # Predict (hasil: 0/1)
        pred = self.classifier.predict([embedding])[0]
        
        return {
            "prediction": int(pred),
            "label": "Qur'an" if pred == 1 else "Non-Qur'an"
        }