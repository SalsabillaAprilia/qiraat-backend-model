#!/usr/bin/env python3
"""Test MFCC extraction with updated preprocessing"""

import sys
sys.path.insert(0, '.')

from utils.audio_utils import extract_mfcc_from_file

# Test dengan WAV file
test_file = './uploads/al-fatihah.wav'
print(f"Testing MFCC extraction with NEW preprocessing:")
print(f"File: {test_file}\n")

features = extract_mfcc_from_file(test_file)

if features is not None:
    print(f"\n✓ SUCCESS!")
    print(f"Feature shape: {features.shape}")
    print(f"Expected: (80,)")
    print(f"Feature vector (first 10): {features[:10]}")
else:
    print("✗ FAILED to extract features")
