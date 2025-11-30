#!/usr/bin/env python3
"""
Advanced offline polyphonic transcription
Features:
  - 4-stem Spleeter separation (vocals, drums, bass, other)
  - Harmonic salience multi-pitch detection (no ML model required)
  - Per-stem multi-track MIDI output
  - Automatic tempo detection & quantization
  - Piano roll PNG export
"""
import os, math, glob
import numpy as np
import librosa, librosa.display
import pretty_midi
import soundfile as sf
import matplotlib.pyplot as plt

# -------------------------
# GLOBALS
# -------------------------
SAMPLE_RATE = 16000
HOP = 512
NFFT = 2048
TOP_N = 8
MIN_NOTE_DURATION = 0.05
MIN_GAP = 0.03

STEMS = ["vocals", "drums", "bass", "other"]
PROGRAMS = {
    "vocals": 52,   # Choir Aahs
    "drums": 0,     # Standard Drum Kit
    "bass": 33,     # Electric Bass (finger)
    "other": 24     # Nylon Guitar or general instrument
}

# -------------------------
# Try to load Spleeter
# -------------------------
try:
    from spleeter.separator import Separator
    SEP = Separator("spleeter:4stems")
except Exception as e:
    SEP = None
    print("⚠️ Spleeter not available. Running WITHOUT stem separation.")
    print("Reason:", e)


# =======================================================
# STEM SEPARATION
# =======================================================
def separate_stems(wav):
    """
    Return dict: {"vocals":path, "bass":path, ...} or {"mix": wav}
    """
    if SEP is None:
        return {"mix": wav}

    base = os.path.splitext(os.path.basename(wav))[0]
    out_dir = f"spleeter_out/{base}"

    try:
        SEP.separate_to_file(wav, "spleeter_out", codec='wav', synchronous=True)
        stems = {}
        for stem in STEMS:
            path = os.path.join(out_dir, f"{stem}.wav")
            if os.path.exists(path):
                stems[stem] = path

        return stems if stems else {"mix": wav}
    except Exception as e:
        print("❌ Spleeter failed:", e)
        return {"mix": wav}


# =======================================================
# POLYPHONIC FREQUENCY ESTIMATION (HARMONIC SALIENCE)
# =======================================================
def harmonic_salience(y, sr):
    """
    Returns:
        times: time per frame
        frames: list of frequency candidates per frame
    """
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT)

    # Emphasize harmonic region
    w = 1 + 0.5 * np.cos(np.linspace(-np.pi, np.pi, len(freqs)))
    S *= w[:, None]

    mag_db = librosa.amplitude_to_db(S, ref=np.max)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)

    frames = []
    for i in range(S.shape[1]):
        mag_col = S[:, i]
        threshold = np.max(mag_col) * 0.12

        idx = np.where(mag_col > threshold)[0]
        candidates = []

        for ix in idx:
            if mag_db[ix, i] < -70:
                continue
            f = freqs[ix]
            if 30 < f < 6000:
                candidates.append(f)

        # Keep strongest TOP_N
        if len(candidates) > TOP_N:
            bins = {f: mag_col[np.argmin(abs(freqs - f))] for f in candidates}
            candidates = sorted(
                candidates,
                key=lambda f: bins[f],
                reverse=True
            )[:TOP_N]

        frames.append(sorted(set(candidates)))

    return times, frames


# =======================================================
# NOTE BUILDING
# =======================================================
def freq_to_midi(f):
    if f <= 0:
        return None
    return int(round(69 + 12 * math.log2(f / 440.0)))


def frame_velocity(frame):
    rms = np.sqrt(np.mean(frame * frame)) + 1e-12
    vel = int(np.clip(30 + 100 * rms / 0.1, 20, 127))
    return vel


def build_notes(times, frames, y, sr):
    active = {}
    notes = []
    frame_len = int(sr * (times[1] - times[0])) if len(times) > 1 else 512

    for i, cand in enumerate(frames):
        midi_set = [freq_to_midi(f) for f in cand if f]
        start = i * HOP
        frame = y[start:start+frame_len]
        vel = frame_velocity(frame)

        # Finish notes that ended
        ended = [m for m in active if m not in midi_set]
        for m in ended:
            s0, l0, vsum, n = active.pop(m)
            s, e = times[s0], times[l0] + (times[1] - times[0])
            if e - s >= MIN_NOTE_DURATION:
                notes.append((m, s, e, int(vsum/n)))

        # Extend or start notes
        for m in midi_set:
            if m in active:
                s0, _, vsum, n = active[m]
                active[m] = (s0, i, vsum + vel, n + 1)
            else:
                active[m] = (i, i, vel, 1)

    # Close remaining
    for m,(s0,l0,vsum,n) in active.items():
        s, e = times[s0], times[l0] + (times[1]-times[0])
        if e - s >= MIN_NOTE_DURATION:
            notes.append((m, s, e, int(vsum/n)))

    notes.sort(key=lambda x: x[1])
    return merge_notes(notes)


def merge_notes(notes):
    merged = []
    by_pitch = {}

    for m, s, e, v in notes:
        by_pitch.setdefault(m, []).append((s, e, v))

    for m, segs in by_pitch.items():
        segs.sort()
        cs, ce, cv = segs[0]
        count = 1

        for s, e, v in segs[1:]:
            if s - ce <= MIN_GAP:
                ce = max(ce, e)
                cv = (cv * count + v) / (count + 1)
                count += 1
            else:
                merged.append((m, cs, ce, int(cv)))
                cs, ce, cv = s, e, v
                count = 1

        merged.append((m, cs, ce, int(cv)))

    merged.sort(key=lambda x: x[1])
    return merged


# =======================================================
# TEMPO & QUANTIZATION
# =======================================================
def detect_tempo(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beats


def quantize_notes(notes, tempo, beats, sr):
    if len(beats) < 2:
        return notes

    beat_times = librosa.frames_to_time(beats, sr=sr)
    grid = np.linspace(beat_times[0], beat_times[-1], len(beat_times) * 4)

    def snap(t):
        return grid[np.argmin(abs(grid - t))]

    qnotes = []
    for m, s, e, v in notes:
        qs, qe = snap(s), snap(e)
        if qe - qs > 0.03:
            qnotes.append((m, qs, qe, v))

    return qnotes


# =======================================================
# MIDI EXPORT
# =======================================================
def save_midi(stem_notes, out_path):
    pm = pretty_midi.PrettyMIDI()

    for stem, notes in stem_notes.items():
        inst = pretty_midi.Instrument(
            program=PROGRAMS.get(stem, 0),
            name=stem
        )
        for m, s, e, v in notes:
            inst.notes.append(pretty_midi.Note(
                velocity=v, pitch=m, start=s, end=e
            ))
        pm.instruments.append(inst)

    pm.write(out_path)


# =======================================================
# PIANO ROLL (PNG)
# =======================================================
def piano_roll_image(pm, out_png):
    fig, ax = plt.subplots(figsize=(12, 4))

    for inst in pm.instruments:
        color = np.random.rand(3,)
        for note in inst.notes:
            ax.plot([note.start, note.end],
                    [note.pitch, note.pitch],
                    lw=4,
                    color=color)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title("Piano Roll")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# =======================================================
# MAIN TRANSCRIPTION PIPELINE
# =======================================================
def transcribe(wav):
    print("🎧 Processing:", wav)

    stems = separate_stems(wav)
    stem_notes = {}

    for stem, path in stems.items():
        print("  → Stem:", stem)

        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        tempo, beats = detect_tempo(y, sr)
        times, frames = harmonic_salience(y, sr)

        notes = build_notes(times, frames, y, sr)
        qnotes = quantize_notes(notes, tempo, beats, sr)

        stem_notes[stem] = qnotes

    base = os.path.splitext(os.path.basename(wav))[0]
    os.makedirs("midi_output", exist_ok=True)

    mid_out = f"midi_output/{base}_poly_multi.mid"
    save_midi(stem_notes, mid_out)

    piano_roll_image(pretty_midi.PrettyMIDI(mid_out),
                     mid_out.replace(".mid", ".png"))

    print("✅ Saved:", mid_out)


def main():
    for wav in sorted(glob.glob("wav_inputs/*.wav")):
        transcribe(wav)


if __name__ == "__main__":
    main()
