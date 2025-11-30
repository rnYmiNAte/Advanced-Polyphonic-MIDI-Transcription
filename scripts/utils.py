#!/usr/bin/env python3
"""
utils.py
Shared helpers for the transcription workflow.
"""

import os


def ensure_dir(path):
    """Make sure a directory exists."""
    os.makedirs(path, exist_ok=True)


def list_audio_files(folder="audio"):
    """Return all MP3 & WAV files in a folder."""
    files = []
    for f in os.listdir(folder):
        if f.lower().ends_with(".mp3") or f.lower().endswith(".wav"):
            files.append(os.path.join(folder, f))
    return files
