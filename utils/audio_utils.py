# utils/audio_utils.py
import numpy as np
import librosa
import os
from pathlib import Path

def extract_mfcc_from_file(filepath, sr=16000, n_mfcc=20, duration=None):
    
    # Check file exists
    if not os.path.exists(filepath):
        print(f"Error extract_mfcc: File not found - {filepath}")
        return None
    
    # Check file size (minimum 100KB to avoid corrupted files)
    file_size = os.path.getsize(filepath)
    if file_size < 100000:
        print(f"Warning: Audio file too small ({file_size} bytes), may be corrupted")
    
    # Try to convert non-WAV formats to WAV if needed
    file_ext = Path(filepath).suffix.lower()
    actual_filepath = filepath
    
    if file_ext in ['.m4a', '.aac', '.mp3', '.flac', '.ogg']:
        # Try to convert to WAV using pydub first, then moviepy as fallback
        wav_path = filepath.rsplit('.', 1)[0] + '.wav'
        
        # Try pydub first (faster)
        conversion_success = False
        try:
            from pydub import AudioSegment
            print(f"Converting {file_ext} to WAV using pydub...")
            audio = AudioSegment.from_file(filepath, format=file_ext[1:])
            audio.export(wav_path, format="wav")
            actual_filepath = wav_path
            conversion_success = True
            print(f"Conversion successful via pydub")
        except ImportError:
            print(f"pydub not available, will try moviepy...")
        except Exception as e:
            print(f"pydub conversion failed: {type(e).__name__} - {str(e)}, trying moviepy...")
        
        # Fallback to moviepy if pydub failed
        if not conversion_success:
            try:
                from moviepy.editor import AudioFileClip
                print(f"Converting {file_ext} to WAV using moviepy...")
                clip = AudioFileClip(filepath)
                clip.audio.write_audiofile(wav_path, verbose=False, logger=None)
                clip.close()
                actual_filepath = wav_path
                conversion_success = True
                print(f"Conversion successful via moviepy")
            except ImportError:
                print(f"moviepy not installed. Install with: pip install moviepy")
            except Exception as e:
                print(f"moviepy conversion failed: {type(e).__name__} - {str(e)}")
        
        if not conversion_success:
            print(f"WARNING: Could not convert {file_ext}. Make sure ffmpeg is installed on your system.")
            print(f"  Windows: choco install ffmpeg")
            print(f"  Mac: brew install ffmpeg")
            print(f"  Linux: apt-get install ffmpeg")
    
    try:
        print(f"Loading audio: {actual_filepath}")
        y, sr_loaded = librosa.load(actual_filepath, sr=sr, duration=duration, mono=True)
        
        if y is None or len(y) == 0:
            print("Error extract_mfcc: Loaded audio is empty")
            return None
        
        print(f"Audio loaded: {len(y)} samples @ {sr_loaded}Hz")
        
        # Extract MFCC (match training: sr=16000, n_mfcc=20)
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr_loaded, 
            n_mfcc=n_mfcc
        )
        
        if mfcc is None or mfcc.size == 0:
            print("Error extract_mfcc: MFCC extraction returned empty")
            return None
        
        print(f"MFCC shape: {mfcc.shape}")
        
        # Compute delta MFCC (1st derivative)
        delta_mfcc = librosa.feature.delta(mfcc)
        print(f"Delta MFCC shape: {delta_mfcc.shape}")
        
        # Compute delta-delta MFCC (2nd derivative)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        print(f"Delta-delta MFCC shape: {delta2_mfcc.shape}")
        
        # Extract statistics (match training)
        mfcc_mean = np.mean(mfcc, axis=1)                    # 20-dim
        mfcc_std = np.std(mfcc, axis=1)                      # 20-dim
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)        # 20-dim
        delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)      # 20-dim
        
        # Concatenate into 80-dimensional feature vector
        # Order MUST match training preprocessing
        features = np.concatenate((
            mfcc_mean,
            mfcc_std,
            delta_mfcc_mean,
            delta2_mfcc_mean
        ))
        
        print(f"Final features shape: {features.shape} (expected: (80,))")
        if features.shape != (80,):
            print(f"WARNING: Feature shape mismatch! Expected (80,), got {features.shape}")
        
        return features
        
    except Exception as e:
        print(f"Error extract_mfcc: {type(e).__name__} - {str(e)}")
        
        # Debug info
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        file_ext = Path(filepath).suffix.lower()
        print(f"\nDEBUG INFO:")
        print(f"  File: {filepath}")
        print(f"  Extension: {file_ext}")
        print(f"  File size: {file_size} bytes")
        
        if file_ext in ['.m4a', '.aac', '.m4b']:
            print(f"\n  NOTE: {file_ext} files require ffmpeg installed!")
            print(f"  To fix, install ffmpeg:")
            print(f"    Windows: choco install ffmpeg (or download from ffmpeg.org)")
            print(f"    Mac: brew install ffmpeg")
            print(f"    Linux: apt-get install ffmpeg")
        
        import traceback
        traceback.print_exc()
        return None
