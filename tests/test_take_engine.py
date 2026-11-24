from frames import hz_at, silence, sung

from sing_pitch.pitch import PitchFrame
from sing_pitch.reference import Band, NearestNote, Target
from sing_pitch.take_engine import TakeEngine, score_take


def live_scores(frames):
    engine = TakeEngine(NearestNote())
    return engine.push(frames)


def test_note_30_cents_sharp_is_yellow():
    scores = live_scores(sung(hz_at(30), 1.0))
    last = scores[-1]
    assert last.band is Band.YELLOW
    assert abs(last.cents - 30) < 0.5
    assert last.note_name == "A4"


def test_note_10_cents_sharp_is_green():
    last = live_scores(sung(hz_at(10), 1.0))[-1]
    assert last.band is Band.GREEN
    assert abs(last.cents - 10) < 0.5


def test_nearest_note_never_goes_red_because_the_note_is_re_chosen():
    # 60 cents flat of A4 is 40 cents sharp of G#4: Nearest note measures against G#4.
    last = live_scores(sung(hz_at(-60), 1.0))[-1]
    assert last.note_name == "G#4"
    assert last.band is Band.YELLOW
    assert abs(last.cents - 40) < 0.5


class FixedNote:
    """Test double Reference: always the same note (what Key and Melody do near their notes)."""

    kind = "melody"

    def __init__(self, midi: float):
        self._midi = midi

    def target(self, t, sung_midi, window_s=0.0):
        return Target(midi=self._midi)


def test_against_a_fixed_note_60_cents_flat_is_red():
    engine = TakeEngine(FixedNote(69.0))
    last = engine.push(sung(hz_at(-60), 1.0))[-1]
    assert last.band is Band.RED
    assert abs(last.cents + 60) < 0.5


def test_silence_between_notes_is_not_drawn_and_not_scored():
    frames = sung(hz_at(0), 0.5) + silence(0.5, start=0.5) + sung(hz_at(0), 0.5, start=1.0)
    scores = live_scores(frames)
    quiet = [s for s in scores if 0.55 < s.t < 0.95]
    assert quiet and all(s.midi is None and s.band is None and s.cents is None for s in quiet)
    assert all(s.band is Band.GREEN for s in scores if s.t < 0.45 or s.t > 1.05)


def run_take(frames, reference=None):
    engine = TakeEngine(reference or NearestNote())
    engine.push(frames)
    return engine.stop()


def test_take_summary_percent_in_tune_and_median_cents_off():
    frames = sung(hz_at(30), 1.0) + silence(1.0, start=1.0) + sung(hz_at(10), 1.0, start=2.0)
    record = run_take(frames)
    s = record.summary
    assert abs(s.pct_in_tune - 50.0) < 0.01          # half the sung time green (+10), half yellow (+30)
    assert abs(s.median_abs_cents - 20.0) < 0.5      # median of |30| and |10| frames
    assert s.tendency == "sharp"
    assert s.lowest_note == s.highest_note == "A4"


def test_silence_does_not_count_as_sung_time():
    frames = sung(hz_at(5), 0.5) + silence(5.0, start=0.5)
    s = run_take(frames).summary
    assert s.pct_in_tune == 100.0
    assert s.n_scored == round(0.5 / 0.016)


def test_a_two_frame_blip_is_neither_scored_nor_counted():
    frames = silence(0.5) + sung(hz_at(45), 0.032, start=0.5) + silence(0.5, start=0.532)
    s = run_take(frames).summary
    assert s.n_scored == 0
    assert s.pct_in_tune is None and s.median_abs_cents is None


def test_stop_keeps_the_raw_pitch_line_untouched():
    frames = sung(hz_at(30), 0.5) + silence(0.5, start=0.5)
    record = run_take(frames)
    assert list(record.frames) == frames
    assert record.reference_kind == "nearest"


def test_rescoring_the_raw_line_gives_the_live_numbers():
    frames = sung(hz_at(30), 1.0) + silence(0.3, start=1.0) + sung(hz_at(-12), 1.0, start=1.3)
    record = run_take(frames)
    assert score_take(record.frames, NearestNote()) == record.summary
