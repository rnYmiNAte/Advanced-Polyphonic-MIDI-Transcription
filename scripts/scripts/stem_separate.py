#!/usr/bin/env python3
"""
stem_separate.py
Runs Spleeter 4-stem separation on an audio file.

Dependencies:
    pip install spleeter soundfile
"""

import os
import sys
import subprocess


def separate_stems(input_path):
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = f"output/stems/{base}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Stem Separation] Processing: {input_path}")
    print(f"[Output] {out_dir}")

    try:
        subprocess.run(
            ["spleeter", "separate", "-p", "spleeter:4stems", "-o", out_dir, input_path],
            check=True
        )
    except FileNotFoundError:
        print("ERROR: Spleeter not installed or not in PATH.")
        sys.exit(1)

    print(f"[OK] Stems created in {out_dir}")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stem_separate.py <audiofile>")
        sys.exit(1)

    separate_stems(sys.argv[1])
