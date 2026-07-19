"""
prepare_pitch_aug.py

Generate formant-preserving pitch-shifted copies of the training audio using the
WORLD vocoder (pyworld). The f0 contour is scaled while the spectral envelope is
kept unchanged, so the timbre identity (formants) is preserved — the model then
learns the target voice across a wider pitch range, which improves quality when
converting with key-shift or outside the dataset's natural range.

Run AFTER resample_normalize_audios.py and BEFORE prepare_data_meta.py, so the
generated copies are picked up by the meta (train split only) and by the feature
extraction scripts.

Output files are named <stem>_pshift{+n}.wav (e.g. song_000_pshift+2.wav);
prepare_data_meta.py keeps files with the '_pshift' marker out of the test split.

Usage:
    python scripts/prepare_pitch_aug.py --data-dir data/finetune --shifts 2,-2
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functools import partial
from pathlib import Path

import click
import numpy as np
import pyworld as pw
import soundfile as sf

from multiprocessing_utils import run_parallel

AUG_MARKER = '_pshift'

F0_FLOOR = 40.0
F0_CEIL = 1100.0


def process_task(task, overwrite, verbose):
    src, semitones = task
    src = Path(src)
    out_path = src.with_name(f"{src.stem}{AUG_MARKER}{semitones:+d}.wav")
    if out_path.exists() and not overwrite:
        if verbose:
            click.echo(f"Skipping existing: {out_path}")
        return
    try:
        x, sr = sf.read(str(src), dtype='float64')
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = np.ascontiguousarray(x)

        f0, t = pw.dio(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL)
        f0 = pw.stonemask(x, f0, t, sr)
        sp = pw.cheaptrick(x, f0, t, sr)
        ap = pw.d4c(x, f0, t, sr)
        # Scale only the pitch; sp (spectral envelope = formants) stays untouched
        y = pw.synthesize(f0 * 2 ** (semitones / 12), sp, ap, sr)

        peak = np.max(np.abs(y))
        if peak > 1.0:
            y = y * (0.99 / peak)
        sf.write(str(out_path), y.astype(np.float32), sr)
        if verbose:
            click.echo(f"Saved: {out_path}")
    except Exception as e:
        click.echo(f"Error processing {src} ({semitones:+d} st): {e}", err=True)


@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Root directory of the preprocessed dataset (data_dir/<speaker>/*.wav).'
)
@click.option(
    '--shifts',
    type=str,
    default='2,-2',
    show_default=True,
    help='Comma-separated semitone shifts to generate (e.g. "2,-2,4,-4").'
)
@click.option(
    '--num-workers',
    type=int,
    default=4,
    show_default=True,
    help='Number of parallel processes (CPU-bound).'
)
@click.option(
    '--overwrite',
    is_flag=True,
    default=False,
    help='Regenerate copies that already exist.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Print per-file logs.'
)
def main(data_dir, shifts, num_workers, overwrite, verbose):
    """Generate formant-preserving pitch-shifted copies of every training wav."""
    try:
        semitone_list = [int(s) for s in shifts.replace(' ', '').split(',') if s]
    except ValueError:
        click.echo(f"Invalid --shifts value: {shifts} (expected e.g. '2,-2,4,-4')", err=True)
        sys.exit(1)
    semitone_list = [s for s in semitone_list if s != 0]
    if not semitone_list:
        click.echo("No non-zero shifts requested, nothing to do.")
        return

    data_dir = Path(data_dir)
    tasks = []
    for speaker_dir in sorted(data_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for wav in sorted(speaker_dir.glob('*.wav')):
            if AUG_MARKER in wav.stem:
                continue  # never re-augment an augmented copy
            tasks.extend((str(wav), s) for s in semitone_list)

    if not tasks:
        click.echo(f"No source wav files found in {data_dir}.", err=True)
        sys.exit(1)

    click.echo(f"Generating {len(tasks)} pitch-shifted copies ({shifts} semitones)...")
    process_func = partial(process_task, overwrite=overwrite, verbose=verbose)
    run_parallel(tasks, process_func, num_workers=num_workers, desc="Pitch-shift augmentation")
    click.echo("Pitch-shift augmentation complete.")


if __name__ == "__main__":
    main()
