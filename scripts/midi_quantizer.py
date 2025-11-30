#!/usr/bin/env python3
"""
midi_quantizer.py
Quantizes a MIDI file to a timing grid.

Dependencies:
    pip install mido numpy
"""

import sys
import os
import mido
import numpy as np

QUANTIZATION = "1/16"   # Options: off, 1/4, 1/8, 1/16

GRID = {
    "off": None,
    "1/4": 120,     # ticks for quarter
    "1/8": 60,
    "1/16": 30
}


def quantize_time(time, grid):
    if grid is None:
        return time
    return int(round(time / grid) * grid)


def quantize_midi(path):
    print(f"[Quantizer] Processing {path}")

    mid = mido.MidiFile(path)
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    grid = GRID.get(QUANTIZATION, None)

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)

        for msg in track:
            if msg.time > 0 and grid is not None:
                msg = msg.copy(time=quantize_time(msg.time, grid))
            new_track.append(msg)

    out = path.replace(".mid", f"_quant.mid")
    new_mid.save(out)
    print(f"[OK] Saved quantized file: {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python midi_quantizer.py <midifile>")
        sys.exit(1)

    quantize_midi(sys.argv[1])
