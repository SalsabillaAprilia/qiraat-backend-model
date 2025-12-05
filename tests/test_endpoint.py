#!/usr/bin/env python3
"""Test Flask endpoint /predict dengan audio file"""

import requests
import os

# Test file yang sudah ada
test_audio = "uploads/al-fatihah.wav"

if not os.path.exists(test_audio):
    print(f"Error: Test audio file not found: {test_audio}")
    exit(1)

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
            print(f"\n✓ SUCCESS! Model accepted 80-dim features")
        else:
            print(f"  Error: {response.text}")
            
except ConnectionRefusedError:
    print("ERROR: Cannot connect to server at http://localhost:5000")
    print("Make sure server is running: python app.py")
except Exception as e:
    print(f"Error: {type(e).__name__} - {str(e)}")
