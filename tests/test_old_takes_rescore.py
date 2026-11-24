"""Takes saved before the song-time log (takes.csv rows with latency_offset_s, npz without song_t) still re-score to
the numbers they were saved with. The fixture was written by the app as it was before ticket #11."""
import csv
import shutil
from pathlib import Path

from melody_helpers import library_song, reference_line

from sing_pitch.take_store import rescore_take, saved_song_times, summary_path

FIXTURE = Path(__file__).parent / "fixtures" / "takes_before_song_time_log"
MELODY = [(-900, 0.6), (-700, 0.6), (-500, 0.6), (-400, 0.6), (-200, 0.6), (-500, 0.6), (-900, 1.0)]
NUMBERS = ("octave_shift", "pct_in_tune", "median_abs_cents", "tendency", "steadiness_cents", "lowest_note",
           "highest_note", "n_voiced_frames", "n_scored_frames", "duration_s", "latency_offset_s")


def old_home(tmp_path):
    shutil.copy(FIXTURE / "takes.csv", tmp_path / "takes.csv")
    shutil.copytree(FIXTURE / "takes", tmp_path / "takes")
    assert library_song(tmp_path, reference_line(MELODY)).id == "do-re-mi"      # the song the Melody Take used
    return tmp_path


def test_old_melody_and_nearest_note_takes_rescore_to_their_saved_numbers(tmp_path):
    home = old_home(tmp_path)
    before = list(csv.DictReader(open(summary_path(home))))
    for row in before:
        rescore_take(home, row["id"], write=True)
    after = list(csv.DictReader(open(summary_path(home))))
    assert [[r[c] for c in NUMBERS] for r in after] == [[r[c] for c in NUMBERS] for r in before]
    assert float(before[0]["pct_in_tune"]) < 99.0          # a non-trivial Take, so equality means something


def test_an_old_melody_take_is_scored_against_t_minus_its_latency_offset(tmp_path):
    from sing_pitch.take_store import load_frames

    home = old_home(tmp_path)
    frames = load_frames(home, "20261120-210000")
    song_times = saved_song_times(home, "20261120-210000")
    assert all(abs(st - (f.t - 0.25)) < 1e-12 for f, st in zip(frames, song_times))
