# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.audio_utils import extract_mfcc_from_file
import numpy as np
import joblib
import time

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow all origins for dev. Lock down in production.

# ---------- load real model + scaler + label encoder ----------
MODEL_JOBLIB = "model/random_forest_model.joblib"
SCALER_JOBLIB = "model/scaler.joblib"
LABEL_ENCODER_JOBLIB = "model/label_encoder.joblib"

model = None
scaler = None
label_encoder = None

if os.path.exists(MODEL_JOBLIB):
    try:
        model = joblib.load(MODEL_JOBLIB)
        print("Loaded model:", MODEL_JOBLIB)
    except Exception as e:
        print("Failed to load model:", e)

if os.path.exists(SCALER_JOBLIB):
    try:
        scaler = joblib.load(SCALER_JOBLIB)
        print("Loaded scaler:", SCALER_JOBLIB)
    except Exception as e:
        print("Failed to load scaler:", e)

if os.path.exists(LABEL_ENCODER_JOBLIB):
    try:
        label_encoder = joblib.load(LABEL_ENCODER_JOBLIB)
        print("Loaded label encoder:", LABEL_ENCODER_JOBLIB)
    except Exception as e:
        print("Failed to load label encoder:", e)

# ------------------ Duration Formatter ------------------
def format_duration(seconds):
    seconds = float(seconds)

    # Jika < 1 detik
    if seconds < 1:
        return f"{seconds:.2f} detik"

    # Jika < 60 detik
    if seconds < 60:
        return f"{seconds:.2f} detik"

    # Jika menit
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes} menit {sec:.0f} detik"

# ---------- Routes ----------
@app.route("/")
def index():
    return {"status": "ok", "msg": "Qiraat dummy API running"}

@app.route("/predict", methods=["POST"])
def predict():

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files['file']
    if f.filename == "":
        return jsonify({"error": "No filename"}), 400

    # save upload temporarily
    save_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(save_path)

    # stopwatch start
    start_time = time.time()

    # extract features
    print(f"\n--- Processing: {f.filename} ---")
    features = extract_mfcc_from_file(save_path)
    if features is None:
        return jsonify({
            "error": "Failed to extract MFCC features from audio file",
            "details": "Supported formats: WAV, MP3, M4A, FLAC, OGG. Check server logs. If using MP3/M4A, ensure ffmpeg is installed on your system."
        }), 500

    # Check if model is loaded
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "details": "Please ensure model files exist in model/ directory"
        }), 500

    # ------------------ Mapping Qiraat & Riwayat ------------------
    QIRAAT_MAP = {
        "warsy": {
            "qiraat": "Nafi'",
            "riwayat": "Warsy"
        },
        "kholaf": {
            "qiraat": "Hamzah",
            "riwayat": "Kholaf"
        }
    } 
    
    # Explanation dictionary for each label
    explanations = {
        "warsy": "Riwayat Warsy 'an Nafi': Pada lafadz 'مالك' dibaca qashr (pendek) menjadi مَلِك (maliki).",
        "kholaf": "Riwayat Kholaf 'an Hamzah: Pada lafadz 'مالك' dibaca qashr (pendek) menjadi مَلِك (maliki), lafadz 'صراط' dibaca isymām (bercampurnya karakter artikulasi ص dengan nuansa ز), dan lafadz 'عليهم' huruf ه dibaca dhammah menjadi عَلَيْهُمْ (ʿalayhum)."
    }

    try:
        # Prepare features for model (2D array: 1 sample, 80 features)
        X = np.array(features).reshape(1, -1)
        
        # Scale features using fitted scaler
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception as e:
                print("Scaler transform error:", e)
                return jsonify({"error": "Feature scaling failed"}), 500

        # Predict using model
        pred_raw = model.predict(X)[0]

        # Get confidence score
        if hasattr(model, "predict_proba"):
            try:
                conf = float(np.max(model.predict_proba(X)))
            except Exception:
                conf = 0.0
        else:
            conf = 0.0

        # Decode prediction to human-readable label using label encoder
        label = None
        try:
            if label_encoder is not None:
                try:
                    label = label_encoder.inverse_transform([pred_raw])[0]
                except Exception:
                    try:
                        label = label_encoder.inverse_transform([int(pred_raw)])[0]
                    except Exception:
                        label = str(pred_raw)
            else:
                label = str(pred_raw)
        except Exception:
            label = str(pred_raw)

        print(f"Prediction: {label}, Confidence: {conf}")
        
        # Get mapping qiraat–riwayat
        mapping = QIRAAT_MAP.get(label, {"qiraat": None, "riwayat": None})

        # Get explanation for the predicted label
        explanation = explanations.get(label, "")

        # stopwatch end
        duration = time.time() - start_time
        readable_latency = format_duration(duration)
        
        return jsonify({
            "prediction": label,
            "qiraat": mapping["qiraat"],
            "riwayat": mapping["riwayat"],
            "confidence": conf,
            "explanation": explanation,
            "latency": readable_latency
        })
        
    except Exception as e:
        print(f"Model predict error: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
