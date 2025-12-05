# Qiraat Backend Model API

Backend Flask untuk prediksi riwayat Qur'an (Qalun, Warsh, Khalaf, Khalad) menggunakan Random Forest dan MFCC features.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Model Files
Pastikan file berikut ada di folder `model/`:
- `random_forest_model.joblib` — trained Random Forest model
- `scaler.joblib` — fitted StandardScaler untuk normalize features
- `label_encoder.joblib` — fitted LabelEncoder untuk decode predictions

Semua file ini akan otomatis dimuat saat server start.

## Running the Server

```bash
python app.py
```

Server akan berjalan di `http://localhost:5000`

Output saat startup (jika model terload):
```
Loaded model: model/random_forest_model.joblib
Loaded scaler: model/scaler.joblib
Loaded label encoder: model/label_encoder.joblib
 * Running on http://0.0.0.0:5000
```

Jika ada file yang tidak ditemukan, server tetap berjalan tapi akan return dummy predictions.

## API Endpoints

### GET `/`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "msg": "Qiraat dummy API running"
}
```

### POST `/predict`
Upload file audio untuk prediksi riwayat.

**Request:**
- Form data dengan key `file` = audio file (format: WAV, MP3, FLAC, dll)

**Response (jika model loaded):**
```json
{
  "prediction": "Qalun",
  "confidence": 0.87,
  "explanation": "Ditemukan sedikit perbedaan durasi pada bacaan hamzah washal..."
}
```

**Response (fallback/dummy):**
```json
{
  "prediction": "Warsh",
  "confidence": 0.65,
  "explanation": "Model mendeteksi variasi panjang vokal dan modulasi nada..."
}
```

## Testing dengan cURL (Windows CMD)

Ubah `C:\path\to\audio.wav` ke path file audio yang sebenarnya:

```cmd
curl -X POST -F "file=@C:\path\to\audio.wav" http://localhost:5000/predict
```

**Contoh response:**
```json
{
  "confidence": 0.92,
  "explanation": "Pola bacaan menunjukkan kecenderungan pendek pada harakat tertentu, khas Riwayat Khalaf.",
  "prediction": "Khalaf"
}
```

## Audio Processing Pipeline

1. **Extract MFCC Features** (`utils/audio_utils.py`)
   - Load audio file dengan librosa (sample rate 22050 Hz)
   - Extract MFCC dengan 13 coefficients
   - Aggregate: mean + std per coefficient → 26-dim feature vector

2. **Scale Features** (jika scaler tersedia)
   - Normalize features menggunakan StandardScaler yang sudah fitted

3. **Predict**
   - Random Forest model memprediksi class index
   - Decode index ke label manusia menggunakan LabelEncoder

4. **Return Result**
   - Prediction (riwayat name)
   - Confidence score (probabilitas tertinggi)
   - Explanation (deskripsi riwayat)

## Notes

- Server automatically saves uploaded files ke folder `uploads/`
- CORS enabled untuk semua origins (development only)
- Error handling untuk file yang rusak atau fitur extraction yang gagal
- Fallback ke dummy prediction jika ada error saat inference

## Troubleshooting

**Server not loading model files?**
- Pastikan path di `app.py` sesuai dengan struktur folder aktual
- Check console output untuk error messages

**Model predictions aneh?**
- Pastikan audio file format compatible (WAV, MP3, FLAC)
- Pastikan model files sesuai dengan feature dimension (26-dim expected)

**Port 5000 sudah terpakai?**
- Edit `app.run(port=5000)` di `app.py` ke port lain, misal 5001
