"""Melody seam: what the reference says at frames the smoothing filled, and where the thresholds fall (gaps bridged under 0.3 s, blips dropped under 0.12 s)."""
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


@pytest.mark.parametrize("frames, filled", [(12, True), (20, True), (29, True), (30, False), (31, False)])
def test_a_dip_of_exactly_300_ms_is_a_stretch_and_a_shorter_one_is_not(frames, filled):
    """An s, t or breath inside a word has no pitch (RMVPE confidence ~0 on TULIP's 0.12-0.3 s gaps, their energy
    centred near 7 kHz against 2.9 kHz for sung frames): bridged, so a phrase is not striped grey (2026-11-25)."""
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


def test_consonant_gaps_inside_a_phrase_leave_no_unscored_stretch_but_a_breath_between_phrases_does():
    """A phrase of four notes split by 0.15, 0.2 and 0.25 s consonants, then a 0.45 s breath, then one more note."""
    notes, gaps = [(0.0, 0.4), (200.0, 0.4), (400.0, 0.4), (200.0, 0.4), (0.0, 0.5)], [15, 20, 25, 45]
    parts = [reference_line([n]) for n in notes]
    t, hz, conf, now = [], [], [], 0.0
    for k, part in enumerate(parts):
        t.append(now + part.t), hz.append(part.hz), conf.append(part.confidence)
        now += float(part.t[-1]) + REF_HOP
        if k < len(gaps):
            g = gaps[k]
            t.append(now + REF_HOP * np.arange(g)), hz.append(np.zeros(g)), conf.append(np.full(g, 0.02))
            now += REF_HOP * g
    line = ReferenceLine(*(np.concatenate(x).astype(np.float32) for x in (t, hz, conf)))
    spans = Melody(line, "x").unscored_spans()
    assert len(spans) == 1
    assert spans[0][1] - spans[0][0] == pytest.approx(0.45, abs=REF_HOP)
