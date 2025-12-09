import librosa
import numpy as np

def load_audio_as_wav(file_path):
    audio = load_audio_16k(file_path)
    return audio, 16000

def preprocess_audio(audio, sr):
    return audio.astype(np.float32)

def load_audio_16k(file_path: str, target_sr: int = 16000):
    
    try:
        # Load audio (librosa default: float32)
        audio, sr = librosa.load(file_path, sr=None, mono=True)

        # Resample jika sample rate bukan 16k
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        return audio.astype(np.float32)

    except Exception as e:
        print(f"[ERROR] Failed to load audio '{file_path}': {e}")
        return None