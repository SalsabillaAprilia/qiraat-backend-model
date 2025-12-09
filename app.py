from flask import Flask, request, jsonify
import os

# ==== Import modul modular ====
from utils.audio_loader import load_audio_as_wav, preprocess_audio
from ml.yamnet_feature_extractor import YamnetFeatureExtractor
from ml.quran_detector import QuranDetector
from ml.qiraat_predictor import QiraatPredictor

# ========== Setup Flask ==========
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== Load Models ==========
yamnet_extractor = YamnetFeatureExtractor()
quran_detector = QuranDetector("model/logistic_regression_model.pkl")
qiraat_predictor = QiraatPredictor("model/random_forest_model.joblib")


# ========== Routing ==========
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "no audio uploaded"}), 400

    file = request.files["audio"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # 1) Load
    audio, sr = load_audio_as_wav(filepath)

    # 2) Preprocess minimal
    audio = preprocess_audio(audio, sr)

    # 3) Extract YAMNet embeddings
    embeddings = yamnet_extractor.extract(audio)

    # 4) Quran / Non-Quran Classification
    is_quran = quran_detector.predict(embeddings)

    if not is_quran:
        return jsonify({
            "result": "Non-Qur'an",
            "detail": "Audio terdeteksi bukan bacaan Qur’an"
        })

    # 5) Qiraat Classification
    qiraat = qiraat_predictor.predict(embeddings)

    return jsonify({
        "result": "Qur'an",
        "qiraat_prediction": qiraat
    })


if __name__ == "__main__":
    app.run(debug=True)