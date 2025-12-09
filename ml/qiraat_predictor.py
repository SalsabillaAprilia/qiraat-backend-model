# ml/qiraat_predictor.py

import numpy as np
import joblib
from utils.audio_utils import extract_mfcc_from_file

class QiraatPredictor:
    def __init__(self,
                 model_path="model/random_forest_model.joblib",
                 scaler_path="model/scaler.joblib",
                 encoder_path="model/label_encoder.joblib"):

        # Load model
        self.model = joblib.load(model_path)

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Load label encoder
        self.label_encoder = joblib.load(encoder_path)

        # Mapping sesuai app.py
        self.QIRAAT_MAP = {
            "warsy": {"qiraat": "Nafi'", "riwayat": "Warsy"},
            "kholaf": {"qiraat": "Hamzah", "riwayat": "Kholaf"},
        }

        # Penjelasan pendek untuk frontend
        self.EXPLANATIONS = {
            "warsy": (
                "Dalam riwayat Warsy 'an Nafi', lafadz 'مالك' "
                "dibaca qashr (pendek) menjadi مَلِك (maliki)."
            ),
            "kholaf": (
                "Dalam riwayat Kholaf 'an Hamzah, lafadz 'مالك' "
                "dibaca qashr (pendek) menjadi مَلِك (maliki), "
                "lafadz 'صراط' dibaca isymām, dan 'عليهم' dibaca dhammah (ʿalayhum)."
            )
        }

    # -------------------------------------------
    # MAIN PREDICT FUNCTION
    # -------------------------------------------
    def predict_qiraat(self, file_path):
        """
        Workflow:
        1. Extract MFCC → 80 fitur
        2. Scale
        3. Predict (raw → label)
        4. Hitung confidence
        5. Return dictionary siap dipakai Flask
        """

        # Step 1: Ekstraksi MFCC
        features = extract_mfcc_from_file(file_path)
        if features is None:
            return {
                "error": True,
                "message": "Gagal ekstrak fitur MFCC dari audio."
            }

        # Step 2: Scale
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Step 3: Predict
        raw_pred = self.model.predict(X_scaled)[0]

        # Step 4: Confidence
        if hasattr(self.model, "predict_proba"):
            confidence = float(np.max(self.model.predict_proba(X_scaled)))
        else:
            confidence = 0.0

        # Step 5: Decode label
        label = self.label_encoder.inverse_transform([raw_pred])[0]

        # Step 6: Qiraat–Riwayat mapping
        mapping = self.QIRAAT_MAP.get(label, {"qiraat": None, "riwayat": None})
        explanation = self.EXPLANATIONS.get(label, "")

        # Return package
        return {
            "error": False,
            "label": label,
            "qiraat": mapping["qiraat"],
            "riwayat": mapping["riwayat"],
            "confidence": confidence,
            "explanation": explanation
        }