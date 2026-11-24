"""Notes mode (#19): the Melody shown and scored as flat notes on exact semitones. The segmentation is a pure function
of the reference line, its scored mask and the Minimum note length. Notes are given as cents from A4 (A4 = MIDI 69):
-500 = E4 (64), -400 = F4 (65), -300 = F#4 (66), -200 = G4 (67), -1200 = A3 (57), -1000 = B3 (59)."""
import math

import numpy as np
import pytest
from melody_helpers import reference_line, with_confidence

from sing_pitch.reference import Melody
from sing_pitch.reference.notes import PER_PHRASE, segment_notes

E4, F4, FS4, G4, A3, B3 = 64, 65, 66, 67, 57, 59


def notes_of(line, min_note_s):
    m = Melody(line, "song")
    return segment_notes(m.line.t, m._midi, m.scored, min_note_s)


def test_a_slide_from_e_to_g_held_mostly_on_g_is_one_g():
    line = reference_line([(-500, 0.1), (-400, 0.1), (-300, 0.1), (-200, 1.0)])     # E F F# slide, then G held
    notes = notes_of(line, 0.5)
    assert [n.midi for n in notes] == [G4]
    assert notes[0].start == 0 and notes[0].end == len(line.t)       # the slide belongs to the G


def test_the_note_is_the_semitone_held_longest_not_the_mean():
    # 0.6 s of E, 0.4 s of G: the mean (65.2) would be F; the most-held semitone is E
    notes = notes_of(reference_line([(-500, 0.6), (-200, 0.4)]), 2.0)
    assert [n.midi for n in notes] == [E4]


def test_a_short_minimum_keeps_a_two_note_line_as_two_notes():
    line = reference_line([(-500, 0.5), (-200, 0.5)])
    notes = notes_of(line, 0.1)
    assert [(n.midi, n.start, n.end) for n in notes] == [(E4, 0, 50), (G4, 50, 100)]


def test_a_blip_inside_a_held_note_leaves_one_note_not_two_of_the_same_pitch():
    line = reference_line([(-500, 0.5), (-400, 0.05), (-500, 0.5), (-200, 0.5)])
    assert [(n.midi, n.start, n.end) for n in notes_of(line, 0.2)] == [(E4, 0, 105), (G4, 105, 155)]


def test_a_long_minimum_merges_them_into_one():
    notes = notes_of(reference_line([(-500, 0.5), (-200, 0.7)]), 2.0)
    assert [n.midi for n in notes] == [G4]


def test_a_note_exactly_as_long_as_the_minimum_is_kept_despite_the_float32_hop():
    # the line's times are float32: a hop of 0.0099999998 s must not make a 30-frame (0.3 s) note shorter than 0.3 s
    line = reference_line([(-500, 1.0), (-200, 0.3), (-500, 1.0)])
    assert [n.midi for n in notes_of(line, 0.3)] == [E4, G4, E4]


def test_one_long_continuous_phrase_of_many_short_runs_segments_in_near_linear_time():
    import time

    rng = np.random.default_rng(0)
    n = 30000                                          # 5 min of 10 ms frames, no pause: one reference Phrase
    semis = np.repeat(60 + rng.integers(0, 12, n), rng.integers(1, 7, n))[:n]       # a run of 1-6 frames, ~8 000 runs
    t0 = time.perf_counter()
    notes = segment_notes(np.arange(n) * 0.01, semis.astype(float), np.ones(n, bool), 0.3)
    assert time.perf_counter() - t0 < 1.5              # the pairwise-rescanning version took ~5 s
    assert notes[0].start == 0 and notes[-1].end == n and all(a.end == b.start for a, b in zip(notes, notes[1:]))


def test_per_phrase_gives_one_note_per_reference_phrase():
    # two Phrases (a 0.8 s pause between them), each with three notes
    first = [(-500, 0.3), (-200, 0.6), (-400, 0.3)]
    second = [(-1200, 0.5), (-1000, 0.2), (-1200, 0.2)]
    line = reference_line(first + [(0, 0.8)] + second)
    line = with_confidence(line, 1.2, 2.0, 0.0)                     # the pause: no reliable reference
    notes = notes_of(line, PER_PHRASE)
    assert [n.midi for n in notes] == [G4, A3]
    assert math.isinf(PER_PHRASE)


def test_a_pause_shorter_than_a_phrase_gap_does_not_split_a_per_phrase_note():
    line = with_confidence(reference_line([(-500, 0.4), (0, 0.3), (-500, 0.4)]), 0.4, 0.7, 0.0)
    assert [n.midi for n in notes_of(line, PER_PHRASE)] == [E4]


def test_notes_never_cover_unscored_stretches():
    line = with_confidence(reference_line([(-500, 1.0), (-200, 1.0)]), 0.8, 1.2, 0.0)
    m = Melody(line, "song")
    notes = m.notes(0.1)
    midi, unscored = notes.sample(np.array([0.5, 1.0, 1.5]))
    assert list(midi[[0, 2]]) == [E4, G4]
    assert np.isnan(midi[1]) and unscored[1]


def test_segmentation_is_deterministic_and_cached_per_song_and_length():
    line = reference_line([(-500, 0.3), (-490, 0.05), (-200, 0.4)])
    a, b = Melody(line, "song").notes(0.2), Melody(line, "song").notes(0.2)
    assert a.note_list == b.note_list
    assert a.note_list is b.note_list                          # second call: the cache
    assert Melody(line, "song").notes(0.1).note_list is not a.note_list


def test_the_notes_melody_targets_the_notes_semitone():
    line = reference_line([(-520, 1.0)])      # a flat E: the line is at 63.8, the note is E4 = 64
    notes = Melody(line, "song").notes(0.3)
    assert notes.kind == "melody" and notes.song_id == "song"
    assert notes.target(0.5, 64.0).midi == E4
    assert Melody(line, "song").target(0.5, 64.0).midi != E4


# --- scoring, the table, re-score ---------------------------------------------------------------------------------
import csv  # noqa: E402

from melody_helpers import library_song, voice_following  # noqa: E402

from sing_pitch.take_engine import LinearClock, TakeEngine, score_take  # noqa: E402
from sing_pitch.take_store import rescore_take, save_take, summary_path  # noqa: E402

FLAT_E = [(-530, 1.5), (-230, 1.5)]       # the singer 30 c flat of E and of G
EXACT = [(-500, 1.5), (-200, 1.5)]


def test_in_notes_mode_the_voice_is_scored_against_the_bars():
    line = reference_line(FLAT_E)
    voice = voice_following(EXACT, 0.0)                     # exactly on E and G
    line_score = score_take(voice, Melody(line, "s")).pct_in_tune
    notes_score = score_take(voice, Melody(line, "s").notes(0.3)).pct_in_tune
    assert line_score < 10 and notes_score > 95


def saved_notes_take(tmp_path, min_note_s):
    song = library_song(tmp_path, reference_line(FLAT_E), seconds=3.0)
    melody = Melody.from_song(song)
    eng = TakeEngine(melody if min_note_s is None else melody.notes(min_note_s), song_clock=LinearClock(0.0))
    eng.push(voice_following(EXACT, 0.0, cents_off=10))
    return save_take(tmp_path, eng.stop())


def row_of(tmp_path, take_id):
    return next(r for r in csv.DictReader(open(summary_path(tmp_path))) if r["id"] == take_id)


def test_a_notes_take_records_its_mode_and_length_and_a_rescore_reproduces_it(tmp_path):
    for length, cell in ((0.3, "0.3"), (PER_PHRASE, "phrase")):
        saved = saved_notes_take(tmp_path, length)
        assert (saved.note_mode, saved.min_note_s) == ("on", length)
        row = row_of(tmp_path, saved.id)
        assert (row["note_mode"], row["min_note_s"]) == ("on", cell)
        again = rescore_take(tmp_path, saved.id)
        assert again.summary == saved.summary
        # 0.3 s: two bars, E and G, both sung; per Phrase: one bar (E and G held equally long: the first, E)
        assert saved.summary.pct_in_tune == (pytest.approx(100.0, abs=3) if length == 0.3 else pytest.approx(50, abs=3))
        assert (again.note_mode, again.min_note_s) == ("on", length)


def test_a_rescore_against_a_given_reference_records_that_references_mode_not_the_rows(tmp_path):
    saved = saved_notes_take(tmp_path, 0.3)
    melody = Melody.from_song(library_song(tmp_path, reference_line(FLAT_E), title="Other", seconds=3.0))
    as_line = rescore_take(tmp_path, saved.id, reference=melody)
    assert (as_line.note_mode, as_line.min_note_s) == ("off", None)
    as_notes = rescore_take(tmp_path, saved.id, reference=melody.notes(PER_PHRASE))
    assert (as_notes.note_mode, as_notes.min_note_s) == ("on", PER_PHRASE)


def test_a_notes_row_with_a_lost_length_stays_notes_at_the_default_length():
    from sing_pitch.library.player_settings import DEFAULT_MIN_NOTE_S
    from sing_pitch.take_store import row_notes

    for cell in ("", "oops", None):
        assert row_notes({"note_mode": "on", "min_note_s": cell}) == ("on", DEFAULT_MIN_NOTE_S)
    assert row_notes({"note_mode": "off", "min_note_s": "0.5"}) == ("off", None)


def test_a_line_take_records_note_mode_off(tmp_path):
    saved = saved_notes_take(tmp_path, None)
    row = row_of(tmp_path, saved.id)
    assert (row["note_mode"], row["min_note_s"]) == ("off", "")
    assert saved.summary.pct_in_tune < 10


def test_old_rows_without_the_columns_load_as_note_mode_off(tmp_path):
    saved = saved_notes_take(tmp_path, None)
    rows = list(csv.DictReader(open(summary_path(tmp_path))))
    with open(summary_path(tmp_path), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[c for c in rows[0] if c not in ("note_mode", "min_note_s")],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    again = rescore_take(tmp_path, saved.id)
    assert again.note_mode == "off" and again.summary == saved.summary


# --- History --------------------------------------------------------------------------------------------------------
from test_history import row as history_row, write_table  # noqa: E402

from sing_pitch.history import load_history  # noqa: E402


def notes_table(home):
    write_table(home, [history_row("a", 1, "melody", "40", song="s"),
                       history_row("b", 2, "melody", "70", song="s", note_mode="on", min_note_s="0.3"),
                       history_row("c", 3, "melody", "80", song="s", note_mode="on", min_note_s="phrase"),
                       history_row("d", 4, "nearest", "55")])


def test_history_filters_by_notes_mode_and_old_rows_count_as_off(tmp_path):
    notes_table(tmp_path)
    h = load_history(tmp_path)
    assert h.note_modes == ["off", "on"]
    assert h.series(note_mode="on").pct_in_tune == [70.0, 80.0]
    assert h.series(note_mode="off").pct_in_tune == [40.0, 55.0]


def test_the_history_view_has_a_notes_mode_filter(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    notes_table(tmp_path)
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    assert v.notes_box.isEnabled()
    v.notes_box.setCurrentIndex(v.notes_box.findData("on"))
    assert list(v.pct_curve.getData()[1]) == [70.0, 80.0]


# --- per-song setting and the window -------------------------------------------------------------------------------
from audio import SR, harmonic, quiet  # noqa: E402
from frames import hz_at  # noqa: E402
from test_centred_playhead import LATENCY, PlayerStub, SONG_S  # noqa: E402

from sing_pitch.audio_input import ArrayInput  # noqa: E402
from sing_pitch.library import Library, PlayerSettings  # noqa: E402
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker  # noqa: E402
from sing_pitch.playback import SongTimeLog  # noqa: E402
from sing_pitch.settings import Settings  # noqa: E402

# a song of 24 one-second notes, each 20 c flat of MIDI 57 + k: the bars sit on 57 + k exactly
SLIGHTLY_FLAT = [(-1220 + 100 * k, 1.0) for k in range(SONG_S)]


def test_notes_mode_and_its_length_are_remembered_per_song(tmp_path):
    a = library_song(tmp_path, reference_line(EXACT), title="A")
    b = library_song(tmp_path, reference_line(EXACT), title="B")
    assert Library(tmp_path).player_settings(a.id).load().notes is False
    Library(tmp_path).player_settings(a.id).save(PlayerSettings(notes=True, min_note_s=PER_PHRASE))
    assert (Library(tmp_path).player_settings(a.id).load().notes,
            Library(tmp_path).player_settings(a.id).load().min_note_s) == (True, PER_PHRASE)
    assert Library(tmp_path).player_settings(b.id).load().notes is False
    import json
    assert json.loads((a.dir / "player.json").read_text())["min_note_s"] == "phrase"     # valid JSON, no Infinity
    (a.dir / "player.json").write_text(json.dumps({"notes": True, "min_note_s": 99}))
    assert Library(tmp_path).player_settings(a.id).load().min_note_s == PlayerSettings().min_note_s


def notes_window(qtbot, tmp_path, length_value, start=True, audio=None):
    """A window with Notes mode ticked at the slider value given (1..40 = tenths of a second, 41 = per Phrase); mid-Take
    when start, the song-time log the test's own (see test_centred_playhead.running)."""
    from sing_pitch.ui import MainWindow

    song = library_song(tmp_path, reference_line(SLIGHTLY_FLAT), seconds=float(SONG_S))
    settings = Settings.load(tmp_path)
    settings.set_latency("WF-1000XM5", LATENCY)
    settings.save()
    log, players = SongTimeLog(SR), []

    def make(path, vocals=None):
        players.append(PlayerStub(path, vocals, log))
        return players[-1]

    win = MainWindow(SwiftF0Tracker(), home=tmp_path,
                     input_factory=lambda _i: ArrayInput(quiet(1.0) if audio is None else audio, SR, 256),
                     backing_factory=make, output_info=lambda: ("WF-1000XM5", "headphones"))
    qtbot.addWidget(win)
    win.reference_box.setCurrentText("Melody")
    win.notes_box.setChecked(True)
    win.note_length_slider.setValue(length_value)
    if start:
        win.toggle()
        qtbot.waitUntil(lambda: hasattr(win._runner.song_clock, "output_at_input_zero_s"), timeout=3000)
        win._timer.stop()
        win._t.clear(), win._midi.clear(), win._band.clear(), win._out.clear(), win._oct.clear()
    return win, song, log


def test_the_controls_name_the_length_and_lock_during_a_take(qtbot, tmp_path):
    win, song, _ = notes_window(qtbot, tmp_path, 41, start=False)
    assert win.note_length_label.text() == "per Phrase"
    win.note_length_slider.setValue(3)
    assert win.note_length_label.text() == "0.3 s"
    saved = Library(tmp_path).player_settings(song.id).load()
    assert (saved.notes, saved.min_note_s) == (True, 0.3)
    win.toggle()
    assert not win.notes_box.isEnabled() and not win.note_length_slider.isEnabled()
    win.toggle()
    assert win.notes_box.isEnabled() and win.note_length_slider.isEnabled()


def test_dragging_the_length_slider_only_moves_the_label_until_it_is_released(qtbot, tmp_path):
    win, song, _ = notes_window(qtbot, tmp_path, 3, start=False)
    previews = []
    win._preview_song = lambda *_: previews.append(1)          # the test's own spy (not on a window method slot)
    win.note_length_slider.setSliderDown(True)
    for v in (4, 5, 6):
        win.note_length_slider.setValue(v)
    assert win.note_length_label.text() == "0.6 s" and previews == []              # the label follows live
    assert Library(tmp_path).player_settings(song.id).load().min_note_s == 0.3     # nothing saved or reloaded yet
    win.note_length_slider.setSliderDown(False)                                    # released
    assert previews == [1]
    assert Library(tmp_path).player_settings(song.id).load().min_note_s == 0.6


def test_the_notes_controls_are_locked_during_a_replay_and_the_preview_follows_them_afterwards(qtbot, tmp_path):
    win, song, _ = notes_window(qtbot, tmp_path, 3, start=False)
    eng = TakeEngine(Melody.from_song(song).notes(0.3), song_clock=LinearClock(0.0))
    eng.push(voice_following(SLIGHTLY_FLAT[:3], 0.0))
    save_take(tmp_path, eng.stop())
    win.toggle_replay()
    assert win._replay is not None
    assert not win.notes_box.isEnabled() and not win.note_length_slider.isEnabled()      # they would change nothing
    previews = []
    win._preview_song = lambda *_: previews.append(1)
    win.toggle_replay()                                                                    # Stop replay
    assert win.notes_box.isEnabled() and win.note_length_slider.isEnabled() and previews == [1]


def visible_labels(win):
    return [(item.toPlainText(), item.pos().x(), item.pos().y()) for item in win._note_labels if item.isVisible()]


def test_live_the_melody_is_drawn_as_labelled_bars_on_exact_semitones_in_both_halves(qtbot, tmp_path):
    win, _, log = notes_window(qtbot, tmp_path, 3)
    log.append(round(10 * SR), 0.0)                          # 10 s rendered from the song start
    offset = win._draw_clock().offset_s
    now = 10.0 + offset - LATENCY
    win._draw_melody(now)
    x, y = win._ref_curve.getData()
    ok = np.isfinite(y)
    assert ok[x < now].mean() > 0.9 and ok[x > now].mean() > 0.9          # past and future halves
    assert np.allclose(y[ok], np.round(y[ok]))                             # exact semitones, not the flat line
    song = x - offset
    inside = ok & (np.abs(song - np.round(song)) > 0.05)
    assert np.allclose(y[inside], 57 + np.floor(song[inside]))
    for curve, d in zip(win._guide_curves, (12, -12)):                     # Guide-line bars an octave either side
        assert np.allclose(curve.getData()[1][ok], y[ok] + d)
    # bars, not a staircase: the line breaks between two notes
    changes = np.flatnonzero(np.diff(np.round(np.nan_to_num(y, nan=-1))) != 0)
    assert all(np.isnan(y[i]) or np.isnan(y[i + 1]) for i in changes)
    labels = visible_labels(win)
    from sing_pitch.music import note_name
    names = [t for t, _, _ in labels]
    assert len(labels) >= 7
    assert note_name(57 + int(np.floor(now - offset))) in names            # the note sounding now
    assert note_name(57 + int(np.floor(now - offset)) + 3) in names        # one in the future half
    for text, lx, ly in labels:
        k = int(np.floor(lx - offset + 0.02))           # each label at the start of its bar (or the plot's left edge)
        assert text == note_name(57 + k) and abs(ly - (57 + k)) < 1.0


def test_an_unscored_dip_inside_a_note_neither_doubles_its_label_nor_trims_the_bar(qtbot, tmp_path):
    win, _, _ = notes_window(qtbot, tmp_path, 3, start=False)
    win._melody = win._reference()
    x = np.arange(0.0, 3.0, 0.01)                                     # notes 57, 58, 59: one second each
    midi, _ = win._melody.sample(x)
    midi[120:140] = np.nan                                            # 1.2-1.4 s unscored, inside the second note
    zeros = np.zeros(len(x))
    out = win._note_bars(x, x, midi, zeros, zeros)
    from sing_pitch.music import note_name
    assert [t for t, _, _ in visible_labels(win)] == [note_name(57), note_name(58), note_name(59)]   # B-flat once
    assert np.isfinite(out[100:120]).all() and np.isnan(out[120:140]).all() and np.isfinite(out[140:190]).all()
    assert np.isnan(out[99]) and np.isfinite(out[90])                 # still broken just before the next note (1.0 s)


def test_label_items_are_reused_not_created_every_frame(qtbot, tmp_path):
    win, _, log = notes_window(qtbot, tmp_path, 3)
    log.append(round(10 * SR), 0.0)
    offset = win._draw_clock().offset_s
    win._draw_melody(10.0 + offset - LATENCY)
    pool = list(win._note_labels)
    for step in range(20):
        win._draw_melody(10.0 + offset - LATENCY - 0.03 * step)
    assert win._note_labels[:len(pool)] == pool and len(win._note_labels) <= len(pool) + 2


def test_a_line_take_draws_no_labels(qtbot, tmp_path):
    win, _, log = notes_window(qtbot, tmp_path, 3, start=False)
    win.notes_box.setChecked(False)
    win.toggle()
    qtbot.waitUntil(lambda: hasattr(win._runner.song_clock, "output_at_input_zero_s"), timeout=3000)
    win._timer.stop()
    log.append(round(10 * SR), 0.0)
    win._draw_melody(10.0)
    assert visible_labels(win) == []
    y = win._ref_curve.getData()[1]
    assert not np.allclose(y[np.isfinite(y)], np.round(y[np.isfinite(y)]))


def test_a_notes_take_is_saved_as_notes_and_replays_as_bars(qtbot, tmp_path):
    audio = np.concatenate([quiet(0.2 + LATENCY)] + [harmonic(hz_at(c + 20), d) for c, d in SLIGHTLY_FLAT[:3]])
    win, song, log = notes_window(qtbot, tmp_path, 41, start=False, audio=audio)
    win.note_length_slider.setValue(5)
    log.append(round(10 * SR), 0.0)
    win.toggle()
    qtbot.waitUntil(lambda: hasattr(win._runner.song_clock, "output_at_input_zero_s"), timeout=3000)
    qtbot.wait(600)
    win.toggle()
    row = next(csv.DictReader(open(summary_path(tmp_path))))
    assert (row["note_mode"], row["min_note_s"]) == ("on", "0.5")
    win.toggle_replay()
    qtbot.waitUntil(lambda: len(win._t) > 40, timeout=3000)
    win.tick()
    y = win._ref_curve.getData()[1]
    assert np.isfinite(y).any() and np.allclose(y[np.isfinite(y)], np.round(y[np.isfinite(y)]))
    assert visible_labels(win)
