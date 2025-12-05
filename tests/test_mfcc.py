#!/usr/bin/env python3
"""Test MFCC extraction with updated preprocessing.

This test will automatically pick an audio file from `uploads/`.
Prefers `.m4a` or `.mp3` if present, otherwise falls back to `.wav`, `.flac`, `.ogg`.
"""

import sys
import glob
import os

sys.path.insert(0, '.')

from utils.audio_utils import extract_mfcc_from_file

search_exts = ('m4a', 'mp3', 'wav', 'flac', 'ogg')
candidates = []
for e in search_exts:
    candidates.extend(glob.glob(f"uploads/*.{e}"))

if not candidates:
    print("Error: No test audio files found in uploads/. Please add one (e.g. sample.m4a)")
    sys.exit(1)

# Prefer an m4a/mp3 if available
preferred = None
for ext in ('m4a', 'mp3'):
    for c in candidates:
        if c.lower().endswith('.' + ext):
            preferred = c
            break
    if preferred:
        break

test_file = preferred or candidates[0]

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
