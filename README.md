# Qiraat Backend Model API

Deskripsi singkat: Backend Flask yang mengimplementasikan sistem identifikasi riwayat bacaan Al-Qur’an berbasis audio menggunakan fitur MFCC dan model Random Forest. Implementasi saat ini difokuskan pada Surat Al-Fatihah dengan dua riwayat, Warsh dan Khalaf, sebagai ruang lingkup awal penelitian.

## Prasyarat
- Python 3.8+
- Install dependency:
```bash
pip install -r requirements.txt
```

## File model (folder `model/`)
Pastikan file berikut tersedia:
- `random_forest_model.joblib`
- `scaler.joblib`
- `label_encoder.joblib`

Model files sudah disertakan di folder `model/`. Server akan memuat model saat startup; jika model gagal dimuat server akan mengembalikan error (500).

## Menjalankan server

```bash
python app.py
```

Server default di http://localhost:5000

## Endpoint singkat
- GET `/` — health check, mengembalikan status singkat.
- POST `/predict` — upload file audio (form key: `file`) untuk mendapat prediksi.

Contoh cURL (Windows CMD):
```cmd
curl -X POST -F "file=@C:\\path\\to\\audio.wav" http://localhost:5000/predict
```

Respon (contoh):
```json
{
   "prediction": ,
   "qiraat": ,
   "riwayat": ,
   "confidence": ,
   "explanation": ,
   "latency": 
}
```

## Ringkasan pipeline
- Ekstraksi fitur audio menggunakan 20 koefisien MFCC beserta turunan delta dan delta-delta, di mana rata-rata dan standar deviasi dari MFCC, serta rata-rata dari delta dan delta-delta, digabungkan menjadi vektor fitur berdimensi 80.
- Standarisasi dengan `scaler.joblib`
- Prediksi dengan `random_forest_model.joblib`
- Decode label dengan `label_encoder.joblib`

## Testing
- Unit tests: jalankan `pytest` pada folder `tests/`:
```bash
python -m pytest tests -q
```

Catatan penting untuk `tests/test_endpoint.py`:
- Skrip pengujian mencari file audio di folder `uploads/` (mis. `uploads/sample.m4a`).
- Sebelum menjalankan pengujian, buat folder `uploads/` di root proyek dan tempatkan file audio uji di sana.

Contoh (Windows CMD / PowerShell):
```bash
mkdir uploads
# salin file audio ke uploads/, mis. uploads/sample.m4a
```

## Catatan singkat
- Uploads sementara disimpan di folder `uploads/` (jika diimplementasikan)
- CORS mungkin diizinkan untuk pengembangan; matikan/konfigurasikan untuk produksi
- Jika model tidak cocok dengan dimensi fitur, training model dan scaler harus konsisten
