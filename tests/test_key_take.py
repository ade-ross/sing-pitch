from frames import hz_at, silence, sung

from sing_pitch.reference import Band, Key
from sing_pitch.take_engine import TakeEngine

# hz_at(cents from A4): C4 = -900, E4 = -500, F4 = -400, F#4 = -300, A4 = 0, Bb4 = +100
C_MAJOR = Key.parse("C major")


def live(frames, key=C_MAJOR):
    return TakeEngine(key).push(frames)


def test_sung_f_sharp_in_c_major_is_flagged_out_of_key():
    last = live(sung(hz_at(-300), 1.0))[-1]
    assert last.in_key is False
    assert last.band is Band.RED          # a semitone from the nearest in-key note


def test_sung_e_15_cents_flat_in_c_major_is_green_and_in_key():
    last = live(sung(hz_at(-515), 1.0))[-1]
    assert last.in_key is True
    assert last.band is Band.GREEN
    assert last.note_name == "E4"
    assert abs(last.cents + 15) < 0.5


def test_in_key_note_is_scored_to_the_nearest_in_key_note_not_the_nearest_chromatic_one():
    # D#4 + 10 cents is out of key (nearest chromatic note D#); it is scored against D or E, never D#.
    last = live(sung(hz_at(-600 + 10), 1.0))[-1]
    assert last.in_key is False
    assert last.note_name in ("D4", "E4")
    assert abs(last.cents) >= 85


def test_note_30_cents_sharp_of_an_in_key_note_is_yellow_and_in_key():
    last = live(sung(hz_at(-500 + 30), 1.0))[-1]
    assert last.in_key is True and last.band is Band.YELLOW


def test_nearest_note_frames_carry_no_in_key_flag():
    from sing_pitch.reference import NearestNote

    assert TakeEngine(NearestNote()).push(sung(hz_at(0), 0.5))[-1].in_key is None


def test_natural_minor_has_its_flat_third_sixth_and_seventh():
    a_minor = Key.parse("A minor")
    # A minor = A B C D E F G: C4 (-900) in key; C#4 (-800) out; G4 (-200) in key, G#4 (-100) out.
    flags = {c: live(sung(hz_at(c), 0.5), a_minor)[-1].in_key for c in (-900, -800, -200, -100)}
    assert flags == {-900: True, -800: False, -200: True, -100: False}


def test_the_key_picker_offers_all_24_keys_with_unique_names():
    keys = Key.all()
    assert len(keys) == 24
    assert len({k.name for k in keys}) == 24
    assert {"C major", "F# major", "Bb minor", "A minor"} <= {k.name for k in keys}


def test_key_names_round_trip_and_flats_and_sharps_both_parse():
    assert Key.parse("Bb minor") == Key.parse("A# minor")
    assert Key.parse("C major").name == "C major"


def test_summary_counts_each_out_of_key_note_once_not_once_per_frame():
    frames = (sung(hz_at(-500), 0.5)                              # E4, in key
              + sung(hz_at(-300), 0.5, start=0.5)                 # F#4, out
              + silence(0.3, start=1.0)
              + sung(hz_at(-100), 0.5, start=1.3)                 # G#4, out
              + sung(hz_at(-500), 0.5, start=1.8))                # E4 again
    eng = TakeEngine(C_MAJOR)
    eng.push(frames)
    record = eng.stop()
    assert record.summary.out_of_key_count == 2
    assert record.key == "C major"
    assert record.reference_kind == "key"


def test_a_clean_in_key_take_has_zero_out_of_key_and_full_in_tune():
    eng = TakeEngine(C_MAJOR)
    eng.push(sung(hz_at(-500 + 5), 1.0) + sung(hz_at(-900 + 5), 1.0, start=1.0))   # E4, then C4
    record = eng.stop()
    assert record.summary.out_of_key_count == 0
    assert record.summary.pct_in_tune == 100.0
