# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.audio_utils import extract_mfcc_from_file
import numpy as np
import joblib  # nanti pakai ini kalau ada model nyata

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow all origins for dev. Lock down in production.

# ---------- Optional: load real model if tersedia ----------
MODEL_PATH = "model/qiraat_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)

# ---------- Routes ----------
@app.route("/")
def index():
    return {"status": "ok", "msg": "Qiraat dummy API running"}

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects form-data with key 'file' = audio file.
    Returns JSON: { prediction: str, confidence: float }
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files['file']
    if f.filename == "":
        return jsonify({"error": "No filename"}), 400

    # save upload temporarily
    save_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(save_path)

    # extract features
    features = extract_mfcc_from_file(save_path)
    if features is None:
        return jsonify({"error": "Failed extract features"}), 500
    explanation = ""
    # If real model loaded -> use it. Otherwise return dummy response.
    if model is not None:
        try:
            # model expects 2D array
            X = np.array(features).reshape(1, -1)
            pred_idx = model.predict(X)[0]
            if hasattr(model, "predict_proba"):
                conf = float(np.max(model.predict_proba(X)))
            else:
                conf = 0.0
            # map label idx to name if needed (assume model outputs strings)
            label = str(pred_idx)
            explanation = dummy_data.get(label, "")
            return jsonify({"prediction": label, "confidence": conf, "explanation": explanation})
        except Exception as e:
            print("Model predict error:", e)
            # fallback to dummy
    # Dummy prediction (use simple heuristic or random)
    dummy_labels = ["Qalun", "Warsh", "Khalaf", "Khalad"]
    # heuristic: use file length as pseudo-feature
    dummy_data = {
            "Qalun": "Ditemukan sedikit perbedaan durasi pada bacaan hamzah washal dan pola nada naik di akhir ayat, khas Riwayat Qalun.",
            "Warsh": "Model mendeteksi variasi panjang vokal dan modulasi nada pada huruf-huruf mad yang khas dalam Riwayat Warsh.",
            "Khalaf": "Pola bacaan menunjukkan kecenderungan pendek pada harakat tertentu, khas Riwayat Khalaf.",
            "Khalad": "Deteksi spektrum menunjukkan penghilangan sebagian getaran hamzah, menandakan pola bacaan Riwayat Khalad.",
        }
    try:
        import soundfile as sf
        info = sf.info(save_path)
        duration = info.duration
    except Exception:
        duration = None

    if duration:
        # pick label based on duration bucket — just to have stable output
        idx = int(duration) % len(dummy_labels)
        label = dummy_labels[idx]
        confidence = round(0.6 + ( (duration - int(duration)) * 0.4 ), 3)
        explanation = dummy_data[label]
    else:
        import random
        label = random.choice(dummy_labels)
        confidence = round(random.uniform(0.5, 0.9), 3)
        explanation = dummy_data[label]

    return jsonify({"prediction": label, "confidence": confidence, "explanation": explanation})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
