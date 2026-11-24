"""Octave shift per Phrase (ticket #17): Auto picks a whole octave for each Phrase, a pinned Octave applies to every
Phrase, old Takes keep their one-shift-per-Take scoring, and the blue line is drawn where the Take is scored."""
import csv
import json

import numpy as np
import pytest
from audio import SR, harmonic, quiet
from frames import HOP, hz_at
from melody_helpers import REF_HOP, library_song
from test_melody_window import window

from sing_pitch.library import Library, PlayerSettings
from sing_pitch.library.ports import ReferenceLine
from sing_pitch.pitch import PitchFrame
from sing_pitch.reference import Band, Melody
from sing_pitch.take_engine import AUTO, LEGACY, LinearClock, TakeEngine, pinned, replay_scores, score_take
from sing_pitch.take_engine.octave import split_phrases
from sing_pitch.take_store import rescore_take, save_take, summary_path

# (cents from A4 or None for a rest, seconds, octaves the voice sings away from the reference)
VERSE = [(-900, 0.6, -1), (-700, 0.6, -1), (-500, 0.6, -1), (-400, 0.6, -1)]
REST = [(None, 1.0, 0)]
CHORUS = [(-200, 0.6, 0), (-500, 0.6, 0), (-900, 0.6, 0), (-700, 0.6, 0), (-500, 0.6, 0)]


def reference_of(segs) -> ReferenceLine:
    t, hz, start = [], [], 0.0
    for cents, seconds, _ in segs:
        if cents is not None:
            n = round(seconds / REF_HOP)
            t += [start + i * REF_HOP for i in range(n)]
            hz += [hz_at(cents)] * n
        start += seconds
    return ReferenceLine(np.array(t, np.float32), np.array(hz, np.float32), np.full(len(t), 0.9, np.float32))


def voice_of(segs) -> list[PitchFrame]:
    """Frames of a voice singing segs, built from the segments (not from the line)."""
    frames, t, end = [], 0.0, sum(s for _, s, _ in segs)
    while t < end:
        edge, hzv = 0.0, None
        for cents, seconds, k in segs:
            if edge <= t < edge + seconds and cents is not None:
                hzv = hz_at(cents + 1200 * k)
            edge += seconds
        frames.append(PitchFrame(t, 0.0, 0.0, -140.0) if hzv is None else PitchFrame(t, hzv, 0.98, -20.0))
        t += HOP
    return frames


TAKE = VERSE + REST + CHORUS


def green_share(scores, lo, hi):
    inside = [x for x in scores if lo <= x.t < hi and x.band is not None]
    return sum(x.band is Band.GREEN for x in inside) / len(inside)


# --- the Phrase -------------------------------------------------------------------------------------------------

def test_a_pause_of_half_a_second_ends_a_phrase_and_a_shorter_one_does_not():
    def runs(*gaps_s):
        mask, frames = [], []
        for g in (0.0,) + gaps_s:
            mask += [False] * round(g / HOP) + [True] * 20
        frames = [PitchFrame(i * HOP, 220.0, 0.9, -20.0) for i in range(len(mask))]
        return frames, mask

    assert len(split_phrases(*runs(0.4))) == 1
    assert len(split_phrases(*runs(0.6))) == 2
    assert len(split_phrases(*runs(0.4, 0.6, 0.3))) == 2


def test_phrases_are_the_same_function_of_frames_whether_they_arrive_at_once_or_in_chunks():
    frames = voice_of(TAKE)
    mask = [f.confidence >= 0.6 for f in frames]
    whole = split_phrases(frames, mask)
    eng = TakeEngine(Melody(reference_of(TAKE), "x"))
    for i in range(0, len(frames), 7):
        eng.push(frames[i : i + 7])
    assert eng.phrase_spans() == whole and len(whole) == 2


# --- Auto: one octave per Phrase --------------------------------------------------------------------------------

def test_a_verse_an_octave_down_and_a_chorus_at_pitch_in_one_take_are_both_in_tune():
    line = reference_of(TAKE)
    frames = voice_of(TAKE)
    s = score_take(frames, Melody(line, "x"))
    assert s.pct_in_tune >= 98.0
    scores = replay_scores(frames, Melody(line, "x"))
    assert green_share(scores, 0.0, 2.4) >= 0.98 and green_share(scores, 3.4, 6.4) >= 0.98
    assert {x.octave for x in scores if x.t < 2.4} == {-1} and {x.octave for x in scores if 4.0 < x.t < 6.0} == {0}


def test_the_summary_octave_shift_is_the_most_common_phrase_shift_ties_going_to_the_more_scored_phrase():
    assert score_take(voice_of(TAKE), Melody(reference_of(TAKE), "x")).octave_shift == 0      # 1 vs 1, chorus is longer


def test_the_summary_octave_shift_names_its_mode_auto():
    s = score_take(voice_of(VERSE + REST + VERSE + REST + CHORUS), Melody(reference_of(VERSE + REST + VERSE + REST + CHORUS), "x"))
    assert (s.octave_shift, s.octave_mode) == (-1, "auto")


def test_one_note_jumping_an_octave_inside_a_phrase_is_still_red():
    segs = [(-900, 1.2, 0), (-900, 0.8, 1)]
    scores = replay_scores(voice_of(segs), Melody(reference_of(segs), "x"))
    assert all(x.band is Band.RED for x in scores if x.t > 1.3 and x.midi is not None)
    assert all(x.band is Band.GREEN for x in scores if x.t < 1.1 and x.midi is not None)


def test_live_the_octave_follows_each_phrase_and_a_growing_phrase_may_be_re_estimated():
    line = reference_of(TAKE)
    frames = voice_of(TAKE)
    eng = TakeEngine(Melody(line, "x"))
    scores = [x for i in range(0, len(frames), 5) for x in eng.push(frames[i : i + 5])]
    assert green_share(scores, 0.5, 2.4) >= 0.98 and green_share(scores, 3.6, 6.4) >= 0.98
    assert {x.octave for x in scores if 1.0 < x.t < 2.4} == {-1} and {x.octave for x in scores if 4.0 < x.t < 6.0} == {0}
    assert eng.stop().summary.pct_in_tune >= 98.0


# --- pinned ----------------------------------------------------------------------------------------------------

def test_a_pinned_octave_applies_to_every_phrase_and_auto_gets_the_per_phrase_choice_back():
    line, frames = reference_of(TAKE), voice_of(TAKE)
    pin = score_take(frames, Melody(line, "x"), octave=pinned(-1))
    assert (pin.octave_shift, pin.octave_mode) == (-1, "pinned")
    scores = replay_scores(frames, Melody(line, "x"), octave=pinned(-1))
    assert green_share(scores, 0.0, 2.4) >= 0.98 and green_share(scores, 3.6, 6.4) <= 0.02
    assert {x.octave for x in scores} == {-1}
    live = TakeEngine(Melody(line, "x"), octave=pinned(-1))
    assert {x.octave for x in live.push(frames)} == {-1}
    assert score_take(frames, Melody(line, "x"), octave=AUTO).pct_in_tune >= 98.0


def test_nearest_note_and_key_takes_have_no_octave_mode():
    from frames import sung

    from sing_pitch.reference import NearestNote

    s = score_take(sung(hz_at(-1200), 1.0), NearestNote(), octave=pinned(-1))
    assert (s.octave_shift, s.octave_mode) == (0, "")


# --- table and re-score ------------------------------------------------------------------------------------------

def saved_take(tmp_path, octave):
    song = library_song(tmp_path, reference_of(TAKE))
    eng = TakeEngine(Melody.from_song(song), song_clock=LinearClock(0.0), octave=octave)
    eng.push(voice_of(TAKE))
    return save_take(tmp_path, eng.stop())


def test_the_octave_mode_is_stored_and_a_rescore_reproduces_both_modes(tmp_path):
    for octave, mode, shift in ((AUTO, "auto", "0"), (pinned(-1), "pinned", "-1")):
        saved = saved_take(tmp_path, octave)
        row = next(r for r in csv.DictReader(open(summary_path(tmp_path))) if r["id"] == saved.id)
        assert (row["octave_mode"], row["octave_shift"]) == (mode, shift)
        assert rescore_take(tmp_path, saved.id).summary == saved.summary


def test_an_old_take_with_no_octave_mode_rescores_with_one_shift_per_take_as_before(tmp_path):
    saved = saved_take(tmp_path, AUTO)
    rows = list(csv.DictReader(open(summary_path(tmp_path))))
    for r in rows:
        r["octave_mode"] = ""                       # a row written before this ticket
    with open(summary_path(tmp_path), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[c for c in rows[0] if c != "octave_mode"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    old = rescore_take(tmp_path, saved.id).summary
    legacy = score_take(voice_of(TAKE), Melody(reference_of(TAKE), "x"), octave=LEGACY)
    assert old.pct_in_tune == pytest.approx(legacy.pct_in_tune) and old.octave_shift == legacy.octave_shift
    assert old.octave_mode == "" and old.pct_in_tune < 70.0       # one k for the whole Take: one half is red
    rescore_take(tmp_path, saved.id, write=True)
    assert next(csv.DictReader(open(summary_path(tmp_path))))["octave_mode"] == ""


# --- per-song setting --------------------------------------------------------------------------------------------

def test_the_octave_is_remembered_per_song_and_defaults_to_auto(tmp_path):
    a = library_song(tmp_path, reference_of(TAKE), title="Song A")
    b = library_song(tmp_path, reference_of(TAKE), title="Song B")
    assert Library(tmp_path).player_settings(a.id).load().octave is None
    Library(tmp_path).player_settings(a.id).save(PlayerSettings(octave=-1))
    assert Library(tmp_path).player_settings(a.id).load().octave == -1
    assert Library(tmp_path).player_settings(b.id).load().octave is None
    (a.dir / "player.json").write_text(json.dumps({"octave": 7}))
    assert Library(tmp_path).player_settings(a.id).load().octave is None


# --- the window --------------------------------------------------------------------------------------------------

def an_octave_down(tmp_path, qtbot, **kw):
    notes = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]
    song = library_song(tmp_path, reference_of([(c, s, 0) for c, s in notes]))
    audio = np.concatenate([quiet(0.2)] + [harmonic(hz_at(c - 1200), d) for c, d in notes] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio, **kw)
    win.reference_box.setCurrentText("Melody")
    return win, song


def test_live_the_blue_line_is_drawn_at_the_voice_octave_inside_the_view_with_guide_lines_an_octave_either_side(
        qtbot, tmp_path):
    win, song = an_octave_down(tmp_path, qtbot)
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 80, timeout=3000)
    win.tick()
    x, blue = win._ref_curve.getData()
    voice = np.nanmedian(np.fromiter(win._midi, float))
    ok = np.isfinite(blue)
    lo, hi = win._plot.viewRange()[1]
    assert lo <= np.nanmedian(blue[ok]) <= hi and abs(np.nanmedian(blue[ok]) - voice) < 4
    assert win._ref_curve.opts["pen"] != win._guide_curves[0].opts["pen"]
    for curve, d in zip(win._guide_curves, (12, -12)):
        gx, gy = curve.getData()
        assert np.allclose(gy[ok], blue[ok] + d)
        assert curve.opts["pen"].color().alpha() < win._ref_curve.opts["pen"].color().alpha() or \
            curve.opts["pen"].widthF() < win._ref_curve.opts["pen"].widthF()
    win.toggle()


def test_the_octave_box_pins_the_take_and_is_remembered_with_the_song(qtbot, tmp_path):
    win, song = an_octave_down(tmp_path, qtbot)
    assert [win.octave_box.itemText(i) for i in range(win.octave_box.count())] == ["Auto", "-2", "-1", "0", "+1"]
    assert win.octave_box.currentData() is None
    win.octave_box.setCurrentIndex(win.octave_box.findData(-1))
    assert Library(tmp_path).player_settings(song.id).load().octave == -1
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 80, timeout=3000)
    assert not win.octave_box.isEnabled()
    win.toggle()
    row = next(csv.DictReader(open(tmp_path / "takes.csv")))
    assert (row["octave_mode"], row["octave_shift"]) == ("pinned", "-1")
    assert "octave shift -1 (pinned)" in win.summary_label.text()
    win.octave_box.setCurrentIndex(0)
    win2, _ = an_octave_down(tmp_path, qtbot)
    assert win2.octave_box.currentData() is None


def test_a_replay_draws_the_blue_line_at_the_octave_the_take_was_scored(qtbot, tmp_path):
    win, song = an_octave_down(tmp_path, qtbot)
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 80, timeout=3000)
    win.toggle()
    win.toggle_replay()
    qtbot.waitUntil(lambda: len(win._t) > 40, timeout=3000)
    win.tick()
    x, blue = win._ref_curve.getData()
    lo, hi = win._plot.viewRange()[1]
    assert lo <= np.nanmedian(blue[np.isfinite(blue)]) <= hi
    assert abs(np.nanmedian(blue[np.isfinite(blue)]) - np.nanmedian(np.fromiter(win._midi, float))) < 4


def test_the_blue_line_before_the_first_phrase_is_drawn_at_the_octave_of_that_phrase(qtbot, tmp_path):
    win, song = an_octave_down(tmp_path, qtbot)
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 80, timeout=3000)
    win.tick()
    t = np.fromiter(win._t, float)
    assert set(win._oct) == {-1} and t[0] < 0.1
    win.toggle()


# --- review fixes (round 3) --------------------------------------------------------------------------------------

def test_a_pinned_octave_never_shifts_a_nearest_note_or_key_take_live():
    from frames import sung

    from sing_pitch.reference import Key, NearestNote

    for ref in (NearestNote(), Key(0, "major")):
        scores = TakeEngine(ref, octave=pinned(-1)).push(sung(hz_at(-1200), 1.0))
        assert {x.octave for x in scores} == {0}
        assert all(abs(x.cents) < 150 for x in scores if x.band is not None and x.cents is not None)


def test_a_pinned_take_still_reports_its_phrases():
    frames = voice_of(TAKE)
    auto, pin = TakeEngine(Melody(reference_of(TAKE), "x")), TakeEngine(Melody(reference_of(TAKE), "x"),
                                                                          octave=pinned(-1))
    auto.push(frames), pin.push(frames)
    assert len(auto.phrase_spans()) == 2 and pin.phrase_spans() == auto.phrase_spans()


def test_a_pinned_row_without_its_shift_rescores_as_an_old_take_not_as_pinned_zero():
    from sing_pitch.take_store import row_octave

    assert row_octave({"octave_mode": "pinned", "octave_shift": ""}) == LEGACY
    assert row_octave({"octave_mode": "pinned", "octave_shift": "abc"}) == LEGACY
    assert row_octave({"octave_mode": "pinned", "octave_shift": "-1"}) == pinned(-1)
    assert row_octave({"octave_mode": "pinned", "octave_shift": "-1.0"}) == pinned(-1)


def test_the_octave_box_does_not_pin_a_take_against_key_or_nearest_note(qtbot, tmp_path):
    win, song = an_octave_down(tmp_path, qtbot)
    win.octave_box.setCurrentIndex(win.octave_box.findData(-1))
    assert win._octave_setting() == pinned(-1)
    for name in ("Key", "Nearest note"):
        if win.reference_box.findText(name) < 0:
            continue
        win.reference_box.setCurrentText(name)
        assert win._octave_setting() == AUTO


@pytest.mark.parametrize("pin", [None, -1])
def test_a_replay_opens_the_view_at_the_octave_the_take_was_scored(qtbot, tmp_path, pin):
    win, song = an_octave_down(tmp_path, qtbot)
    if pin is not None:
        win.octave_box.setCurrentIndex(win.octave_box.findData(pin))
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 80, timeout=3000)
    win.toggle()
    win.octave_box.setCurrentIndex(win.octave_box.findData(0))       # the box now says something else
    ref_midi = np.nanmedian(win._melody.midi_line()[1])
    win.toggle_replay()
    assert abs(win._centre - (ref_midi - 12)) < 1.0
    win.toggle_replay()


def test_the_octave_box_choices_come_from_the_player_settings(qtbot, tmp_path, monkeypatch):
    from sing_pitch.library import player_settings

    monkeypatch.setattr(player_settings, "OCTAVE_CHOICES", (-1, 0))
    win, _ = an_octave_down(tmp_path, qtbot)
    assert [win.octave_box.itemText(i) for i in range(win.octave_box.count())] == ["Auto", "-1", "0"]
