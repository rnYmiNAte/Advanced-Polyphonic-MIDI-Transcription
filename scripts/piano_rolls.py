#!/usr/bin/env python3
"""
piano_roll.py
Generates piano-roll images from MIDI files.

Dependencies:
    pip install mido numpy matplotlib
"""

import os
import sys
import mido
import numpy as np
import matplotlib.pyplot as plt


def midi_to_pianoroll(mid, resolution=0.01):
    """ Convert MIDI to (note × time) piano roll matrix """

    max_note = 108
    min_note = 21
    note_range = max_note - min_note + 1

    # Track total time
    total_time = 0
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
        total_time = max(total_time, t)

    total_seconds = total_time / (mid.ticks_per_beat * 2)  # 120 BPM

    frames = int(total_seconds / resolution) + 1
    roll = np.zeros((note_range, frames))

    for track in mid.tracks:
        time = 0
        active = {}

        for msg in track:
            time += msg.time
            sec = time / (mid.ticks_per_beat * 2)
            frame = int(sec / resolution)

            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = frame

            elif msg.type in ["note_off", "note_on"] and msg.note in active:
                start = active[msg.note]
                end = frame
                roll[msg.note - min_note, start:end] = 1
                del active[msg.note]

    return roll


def plot_pianoroll(roll, out_path):
    plt.figure(figsize=(12, 6))
    plt.imshow(roll, aspect="auto", origin="lower")
    plt.xlabel("Time")
    plt.ylabel("MIDI Note")
    plt.title("Piano Roll")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved piano roll → {out_path}")


def convert_midi_to_png(path):
    mid = mido.MidiFile(path)
    roll = midi_to_pianoroll(mid)

    out_dir = "output/piano_roll"
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{base}.png")

    plot_pianoroll(roll, out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python piano_roll.py <midifile>")
        sys.exit(1)

    convert_midi_to_png(sys.argv[1])
