"""Melody seam: what the reference says at frames the smoothing filled, and where the 0.12 s thresholds fall."""
import numpy as np
import pytest
from frames import hz_at
from melody_helpers import REF_HOP, reference_line

from sing_pitch.library.ports import ReferenceLine
from sing_pitch.reference import Melody
from sing_pitch.reference.melody import scored_mask

A4_MIDI = 69


def line_with_gap(gap_frames: int, junk_hz, first_cents=0.0, second_cents=200.0, note_s=0.5):
    """Two notes with `gap_frames` low-confidence frames between them whose hz is `junk_hz` (what RMVPE may give)."""
    a = reference_line([(first_cents, note_s)])
    b = reference_line([(second_cents, note_s)])
    n = gap_frames
    t = np.concatenate([a.t, a.t[-1] + REF_HOP * (1 + np.arange(n)), a.t[-1] + REF_HOP * (n + 1 + np.arange(len(b.t)))])
    hz = np.concatenate([a.hz, np.full(n, junk_hz), b.hz])
    conf = np.concatenate([a.confidence, np.full(n, 0.05), b.confidence])
    return ReferenceLine(t.astype(np.float32), hz.astype(np.float32), conf.astype(np.float32)), a.t[-1]


@pytest.mark.parametrize("junk_hz", [0.0, 1.0, 4000.0, float("nan")])
def test_a_filled_gap_is_interpolated_between_its_reliable_neighbours_never_taken_from_its_own_hz(junk_hz):
    line, last_of_a = line_with_gap(5, junk_hz)               # 50 ms: smoothed into the note, not an Unscored stretch
    melody = Melody(line, "x")
    mid = float(last_of_a) + 3 * REF_HOP
    target = melody.target(mid, 0.0)
    assert target.midi is not None
    assert A4_MIDI - 0.01 <= target.midi <= A4_MIDI + 2.01      # between A4 and B4 (+200 cents), nowhere near -275
    t, midi = melody.midi_line()
    assert np.isfinite(midi).all() and midi.min() >= A4_MIDI - 0.01 and midi.max() <= A4_MIDI + 2.01


def test_the_midpoint_of_a_filled_gap_is_halfway_between_the_neighbours():
    line, last_of_a = line_with_gap(5, 0.0)
    target = Melody(line, "x").target(float(last_of_a) + 3 * REF_HOP, 0.0)
    assert target.midi == pytest.approx(A4_MIDI + 1.0, abs=0.01)


def test_unreliable_frames_at_the_edge_of_the_line_hold_the_nearest_reliable_pitch():
    line = reference_line([(0.0, 1.0)])
    hz, conf = line.hz.copy(), line.confidence.copy()
    hz[:10] = 0.0
    conf[:10] = 0.05
    melody = Melody(ReferenceLine(line.t, hz, conf), "x", scored=np.ones(len(hz), bool))
    _, midi = melody.midi_line()
    assert midi[:10] == pytest.approx([A4_MIDI] * 10, abs=0.01)


@pytest.mark.parametrize("frames, filled", [(11, True), (12, False), (13, False)])
def test_a_dip_of_exactly_120_ms_is_a_stretch_and_a_shorter_one_is_not(frames, filled):
    line, _ = line_with_gap(frames, 440.0, second_cents=0.0)
    mask = scored_mask(line)
    gap = mask[50 : 50 + frames]
    assert bool(gap.all()) is filled


@pytest.mark.parametrize("frames, kept", [(11, False), (12, True)])
def test_a_reliable_run_of_exactly_120_ms_is_scored_and_a_shorter_one_is_a_blip(frames, kept):
    base = reference_line([(0.0, 1.0)])
    conf = np.full(len(base.t), 0.05, np.float32)
    conf[40 : 40 + frames] = 0.9
    mask = scored_mask(ReferenceLine(base.t, base.hz, conf))
    assert bool(mask[40 : 40 + frames].all()) is kept
