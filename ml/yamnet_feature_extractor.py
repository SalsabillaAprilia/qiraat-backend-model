import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class YamnetFeatureExtractor:
    def __init__(self, model_path: str = None):
        # Load model dari TensorFlow Hub kalau path tidak diberikan
        if model_path is None:
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        else:
            self.model = tf.keras.models.load_model(model_path)

    def extract_embedding(self, audio_waveform: np.ndarray, sample_rate: int = 16000):
        # Pastikan format float32
        audio_waveform = audio_waveform.astype(np.float32)

        # YAMNet expect shape (n_samples,)
        scores, embeddings, spectrogram = self.model(audio_waveform)

        # Ambil rata-rata embedding biar fix-size
        embedding_mean = np.mean(embeddings, axis=0)

        return embedding_mean