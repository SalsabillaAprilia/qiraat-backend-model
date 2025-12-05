#!/usr/bin/env python3
"""Test Flask endpoint /predict dengan audio file"""

import requests
import os

# Find a test audio file in `uploads/` (prefer .m4a/.mp3)
import glob

search_exts = ('m4a', 'mp3', 'wav', 'flac', 'ogg')
candidates = []
for e in search_exts:
    candidates.extend(glob.glob(f"uploads/*.{e}"))

if not candidates:
    print("Error: No test audio files found in uploads/. Please add one (e.g. sample.m4a)")
    exit(1)

# Prefer an m4a if available
preferred = None
for ext in ('m4a', 'mp3'):
    for c in candidates:
        if c.lower().endswith('.' + ext):
            preferred = c
            break
    if preferred:
        break

test_audio = preferred or candidates[0]
print(f"Using test audio: {test_audio}")

# URL endpoint
url = "http://localhost:5000/predict"

print(f"Testing endpoint: {url}")
print(f"Test file: {test_audio}\n")

try:
    # Prepare request
    with open(test_audio, 'rb') as f:
        files = {'file': (os.path.basename(test_audio), f)}
        
        print("Sending POST request...")
        response = requests.post(url, files=files)
        
        print(f"Status code: {response.status_code}")
        print(f"Response:\n")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Prediction: {result.get('prediction')}")
            print(f"  Confidence: {result.get('confidence')}")
            print(f"  Explanation: {result.get('explanation')}")
            print(f"  Latency: {result.get('latency')}")
            print(f"\n✓ SUCCESS! Model accepted 80-dim features")
        else:
            print(f"  Error: {response.text}")
            
except ConnectionRefusedError:
    print("ERROR: Cannot connect to server at http://localhost:5000")
    print("Make sure server is running: python app.py")
except Exception as e:
    print(f"Error: {type(e).__name__} - {str(e)}")
