# utils/audio_utils.py
import numpy as np
import librosa
import os
import tempfile
import shutil


class AudioValidationError(Exception):
    """Raised when audio fails validation (e.g., too long)."""
    pass

def check_ffmpeg_installed():
    if shutil.which("ffmpeg") is None:
        return False
    return True

def extract_mfcc_from_bytes(audio_bytes, sr=16000, n_mfcc=20, duration=None, max_duration_seconds=120):
    if not check_ffmpeg_installed():
        print("FFMPEG is not installed or not found in PATH.")
        return None

    if audio_bytes is None or len(audio_bytes) == 0:
        print("MFCC extraction ERROR: Empty audio bytes")
        return None

    # Quick format detection by header bytes to choose an appropriate suffix
    header = audio_bytes[:12]
    if header.startswith(b'RIFF'):
        suffix = '.wav'
    elif header.startswith(b'ID3') or (len(audio_bytes) > 2 and audio_bytes[0] == 0xff and (audio_bytes[1] & 0xe0) == 0xe0):
        suffix = '.mp3'
    elif header.startswith(b'OggS'):
        suffix = '.ogg'
    else:
        suffix = '.bin'

    tmp_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # If the bytes are already WAV, avoid conversion and load directly
        if suffix == '.wav':
            try:
                y, sr_loaded = librosa.load(tmp_path, sr=sr, duration=duration, mono=True)
            except Exception as e:
                print("MFCC extraction ERROR (direct load):", e)
                return None
        else:
            wav_path = tmp_path + '.wav'
            conversion_success = False
            try:
                from pydub import AudioSegment
                fmt = suffix.lstrip('.') if suffix != '.bin' else None
                print(f"Converting to WAV using pydub (fmt={fmt})...")
                if fmt:
                    audio = AudioSegment.from_file(tmp_path, format=fmt)
                else:
                    audio = AudioSegment.from_file(tmp_path)
                audio.export(wav_path, format='wav')
                conversion_success = True
                print("Conversion OK (pydub)")
            except Exception as e:
                print("pydub failed:", e)

            if not conversion_success:
                try:
                    from moviepy.editor import AudioFileClip
                    print("Converting to WAV via moviepy...")
                    clip = AudioFileClip(tmp_path)
                    clip.audio.write_audiofile(wav_path, verbose=False, logger=None)
                    clip.close()
                    conversion_success = True
                    print("Conversion OK (moviepy)")
                except Exception as e:
                    print("moviepy failed:", e)

            if not conversion_success:
                print("FFMPEG conversion FAILED")
                return None

            try:
                y, sr_loaded = librosa.load(wav_path, sr=sr, duration=duration, mono=True)
            except Exception as e:
                print("MFCC extraction ERROR (after conversion):", e)
                return None

        if y is None or len(y) == 0:
            print("MFCC extraction ERROR: audio empty after load")
            return None

        # validate duration (seconds)
        try:
            actual_duration = librosa.get_duration(y=y, sr=sr_loaded)
        except Exception as e:
            print("Could not determine duration:", e)
            actual_duration = None

        if actual_duration is not None and actual_duration > max_duration_seconds:
            print(f"Audio validation failed: duration {actual_duration:.2f}s > {max_duration_seconds}s")
            raise AudioValidationError(f"Audio duration too long: {actual_duration:.2f}s (max {max_duration_seconds}s)")

        mfcc = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        delta_mean = np.mean(delta, axis=1)
        delta2_mean = np.mean(delta2, axis=1)

        features = np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta2_mean])

        if features.shape != (80,):
            print("WARNING: feature shape mismatch:", features.shape)

        return features
    finally:
        # Cleanup temp files reliably
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass