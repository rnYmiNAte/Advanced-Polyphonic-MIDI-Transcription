#!/usr/bin/env python3
"""
polyphonic_midi_plus.py
Lightweight polyphonic audio → MIDI transcription.

Dependencies:
    pip install numpy scipy mido soundfile

Works with:
    - WAV (PCM)
    - MP3 (via soundfile → ffmpeg backend)
"""

import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq
from mido import Message, MidiFile, MidiTrack

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
FRAME_SIZE = 4096
HOP = 512
PEAK_THRESHOLD_DB = -45        # Peak threshold (dB)
MAX_NOTES_PER_FRAME = 6        # Max polyphony per frame
MIN_NOTE_LENGTH = 5            # Minimum number of frames to count as a note
SAMPLE_RATE = 44100            # Resampled target (soundfile auto)
MIDI_VELOCITY = 90


# -----------------------------------------------------------
# UTILS
# -----------------------------------------------------------
def hz_to_midi(hz):
    if hz <= 0:
        return None
    return 69 + 12 * np.log2(hz / 440.0)


def midi_to_hz(note):
    return 440.0 * (2 ** ((note - 69) / 12))


# -----------------------------------------------------------
# 1. LOAD AUDIO
# -----------------------------------------------------------
def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        print(f"[WARN] Input sample rate = {sr}. Resampling recommended.")
    return audio, sr


# -----------------------------------------------------------
# 2. POLYPHONIC FFT PEAK DETECTION
# -----------------------------------------------------------
def frame_audio(audio, frame_size, hop):
    for i in range(0, len(audio) - frame_size, hop):
        yield audio[i:i + frame_size]


def detect_peaks(mag, freqs):
    """
    Detect strongest harmonic peaks up to MAX_NOTES_PER_FRAME.
    """
    # Convert to dB
    db = 20 * np.log10(np.maximum(mag, 1e-12))

    # Threshold
    mask = db > PEAK_THRESHOLD_DB
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return []

    # Sort peaks by magnitude
    peak_indices = np.argsort(mag[idx])[-MAX_NOTES_PER_FRAME:]
    strongest = idx[peak_indices]

    frequencies = freqs[strongest]
    frequencies = sorted(frequencies)

    return frequencies


# -----------------------------------------------------------
# 3. FRAME → MULTI-PITCH TRACKING
# -----------------------------------------------------------
def extract_multi_pitch(audio, sr):
    window = get_window('hann', FRAME_SIZE)
    freqs = rfftfreq(FRAME_SIZE, 1 / sr)

    chroma_frames = []

    for frame in frame_audio(audio, FRAME_SIZE, HOP):
        frame = frame * window
        spec = rfft(frame)
        mag = np.abs(spec)

        peaks = detect_peaks(mag, freqs)
        chroma_frames.append(peaks)

    return chroma_frames


# -----------------------------------------------------------
# 4. NOTE ONSET / OFFSET TRACKING
# -----------------------------------------------------------
def track_notes(frames):
    """
    frames = list of frequency lists per frame
    returns list of (midi_note, start_time, end_time)
    """
    notes = {}       # midi → (start_frame, last_seen_frame)
    output = []

    for i, freqs in enumerate(frames):
        active_midi = [round(hz_to_midi(f)) for f in freqs if hz_to_midi(f) is not None]

        # Handle note off events
        finished = []
        for midi, (start, last) in notes.items():
            if midi not in active_midi:
                # Note ended
                if (i - start) >= MIN_NOTE_LENGTH:
                    output.append((midi, start, last))
                finished.append(midi)

        for f in finished:
            del notes[f]

        # Handle note on or continuation
        for midi in active_midi:
            if midi not in notes:
                notes[midi] = (i, i)
            else:
                start, _ = notes[midi]
                notes[midi] = (start, i)

    # Handle trailing notes
    for midi, (start, last) in notes.items():
        if (last - start) >= MIN_NOTE_LENGTH:
            output.append((midi, start, last))

    return output


# -----------------------------------------------------------
# 5. WRITE MIDI FILE
# -----------------------------------------------------------
def write_midi(notes, sr, hop, out_path):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat = 480
    mid.ticks_per_beat = ticks_per_beat
    sec_per_tick = 1 / (120 * ticks_per_beat)  # 120 BPM

    # Convert frames → time
    for midi, start, end in notes:
        start_sec = (start * hop) / sr
        end_sec = (end * hop) / sr

        start_ticks = int(start_sec / sec_per_tick)
        end_ticks = int(end_sec / sec_per_tick)

        track.append(Message("note_on", note=midi, velocity=MIDI_VELOCITY, time=start_ticks))
        track.append(Message("note_off", note=midi, velocity=0, time=end_ticks - start_ticks))

    mid.save(out_path)
    print(f"[OK] Saved MIDI → {out_path}")


# -----------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------
def transcribe(input_path):
    print(f"\n=== Polyphonic Transcription ===")
    print(f"Input: {input_path}")

    audio, sr = load_audio(input_path)

    print("[1] Multi-pitch extraction...")
    frames = extract_multi_pitch(audio, sr)

    print("[2] Tracking notes...")
    notes = track_notes(frames)

    print(f"[3] Exporting MIDI... ({len(notes)} notes)")

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = "output/midi"
    os.makedirs(out_dir, exist_ok=True)

    out = os.path.join(out_dir, f"{base}.mid")
    write_midi(notes, sr, HOP, out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python polyphonic_midi_plus.py <audiofile>")
        sys.exit(1)

    transcribe(sys.argv[1])
