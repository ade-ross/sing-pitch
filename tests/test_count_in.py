"""Ticket #42: Count-in between Passes, in the Transport's pure core. One bar of metronome clicks at the song's tempo
(scaled by Speed) replaces the Lead-in; the song is silent during it and starts exactly at the Section start. The song
is a ramp (frame i holds i + 1000), so any song audio is far louder than any click and shows where it came from."""
import math

import numpy as np
import pytest

from sing_pitch.playback.transport import CountIn, Loop, Transport

SR = 8000
SECTION = (3.0, 5.0)
TEMPO = CountIn(bpm=120.0, beats_per_bar=4)      # a beat is 0.5 s of song, a bar 2.0 s


def ramp_song(seconds: float, speed: float = 1.0) -> np.ndarray:
    n = round(seconds * SR / speed)
    return np.stack([np.arange(n, dtype=np.float32) * speed + 1000.0] * 2, axis=1)


def onsets(audio: np.ndarray) -> list[int]:
    """Output frames where a click starts: sound after at least 50 ms of silence."""
    loud = np.abs(audio[:, 0]) > 1e-4
    gap = round(0.05 * SR)
    return [i for i in np.flatnonzero(loud) if not loud[max(0, i - gap):i].any()]


def render_first_pass(speed: float, count_in: CountIn = TEMPO, section=SECTION):
    tr = Transport(ramp_song(10.0), sample_rate=SR)
    if speed != 1.0:
        tr.set_speed(speed, ramp_song(10.0, speed))
    tr.set_loop(Loop(*section, "Verse", count_in=count_in))
    blocks = [tr.render(256) for _ in range(round(4.5 * SR / speed / 256))]
    return tr, blocks


@pytest.mark.parametrize("speed", [1.0, 0.7])
def test_one_bar_of_clicks_ends_one_beat_before_the_section_start_and_the_song_starts_exactly_there(speed):
    _, blocks = render_first_pass(speed)
    audio = np.concatenate([b.audio for b in blocks])
    song_t = np.concatenate([b.song_t for b in blocks])
    beat_out = 0.5 / speed                                     # a beat in output seconds: clicks scale with Speed
    section_out = math.ceil(4 * beat_out * SR - 1e-9)          # the first frame at or after one bar from the Pass start
    clicks = onsets(audio[:section_out])
    assert len(clicks) == 4
    assert np.allclose(np.array(clicks) / SR, [0, beat_out, 2 * beat_out, 3 * beat_out], atol=1.5 / SR)
    assert abs((section_out - clicks[-1]) / SR - beat_out) < 1.5 / SR       # the last click: one beat before
    # silence between the clicks: no song is heard in the count-in (song audio is >= 1000)
    assert np.abs(audio[:section_out]).max() < 1.0
    # the song resumes at the Section start, at that very output frame
    assert np.isclose(audio[section_out, 0], 1000.0 + SECTION[0] * SR, atol=speed + 0.01)
    assert np.isclose(song_t[section_out], SECTION[0], atol=1.5 / SR)
    assert np.all(song_t[:section_out] < SECTION[0])          # the count-in is song time before the Section


def test_the_first_beat_of_the_bar_is_accented():
    _, blocks = render_first_pass(1.0)
    audio = np.concatenate([b.audio for b in blocks])[: 2 * SR]
    beat = SR // 2
    peaks = [np.abs(audio[k * beat: k * beat + beat // 2, 0]).max() for k in range(4)]
    assert peaks[0] > 1.3 * max(peaks[1:]) and min(peaks[1:]) > 0.05


def test_beats_per_bar_sets_the_number_of_clicks():
    _, blocks = render_first_pass(1.0, CountIn(bpm=90.0, beats_per_bar=3))
    audio = np.concatenate([b.audio for b in blocks])
    bar_out = round(3 * (60 / 90) * SR)
    assert len(onsets(audio[:bar_out])) == 3
    assert np.isclose(audio[bar_out, 0], 1000.0 + SECTION[0] * SR)


def test_every_pass_starts_with_the_count_in_again():
    tr = Transport(ramp_song(10.0), sample_rate=SR)
    tr.set_loop(Loop(*SECTION, count_in=TEMPO))
    events, audio = [], []
    for _ in range(round(8.5 * SR / 256)):          # a Pass = 2 s count-in + 2 s Section
        b = tr.render(256)
        events += b.events
        audio.append(b.audio)
    audio = np.concatenate(audio)
    starts = [e for e in events if e.kind == "pass_started"]
    assert [e.pass_index for e in starts] == [1, 2, 3]
    assert np.allclose([e.output_s for e in starts], [0.0, 4.0, 8.0], atol=1 / SR)
    assert np.allclose([e.song_t for e in starts], SECTION[0] - 2.0)
    assert len(onsets(audio[4 * SR: 6 * SR])) == 4 and np.abs(audio[4 * SR: 6 * SR]).max() < 1.0


def test_a_count_in_longer_than_the_song_before_the_section_still_plays_a_whole_bar():
    """Section at 1 s, bar 2 s: the count-in is song time -1 .. 1 (before the song start), never the song itself."""
    _, blocks = render_first_pass(1.0, section=(1.0, 3.0))
    audio = np.concatenate([b.audio for b in blocks])
    song_t = np.concatenate([b.song_t for b in blocks])
    assert len(onsets(audio[: 2 * SR])) == 4 and np.abs(audio[: 2 * SR]).max() < 1.0
    assert np.isclose(song_t[0], -1.0) and np.isclose(audio[2 * SR, 0], 1000.0 + 1.0 * SR)


def test_without_a_count_in_the_lead_in_plays_the_song_as_before():
    tr = Transport(ramp_song(10.0), sample_rate=SR)
    tr.set_loop(Loop(*SECTION, lead_in_s=2.0))
    b = tr.render(256)
    assert np.isclose(b.audio[0, 0], 1000.0 + 1.0 * SR)


@pytest.mark.parametrize("section", [SECTION, (0.5, 2.0)])     # the second Count-in starts before the song start
def test_a_speed_change_asked_with_the_loop_crossfades_against_silence_in_the_count_in(section):
    """Review 5 #2: a Speed change pending when the first Pass starts used to crossfade the old Speed's song into the
    Count-in, and a Count-in before the song start (negative song frame) wrapped the slice to the song end."""
    tr = Transport(ramp_song(10.0), sample_rate=SR)
    tr.render(256)                                  # something rendered: a Speed change now crossfades
    tr.set_speed(0.7, ramp_song(10.0, 0.7))
    tr.set_loop(Loop(*section, count_in=TEMPO))
    audio = np.concatenate([tr.render(256).audio for _ in range(round(1.5 * SR / 256))])
    assert np.abs(audio).max() < 1.0                # only clicks: the song (>= 1000) is never heard



def test_changing_only_the_between_passes_keeps_the_pass_and_applies_at_the_next_wrap():
    """Review 5 #5: the tempo found (or the BPM, beats or Between changed) while looping used to restart at Pass 1.
    The same Section with another Lead-in or Count-in goes on; the next Pass starts with the new one."""
    tr = Transport(ramp_song(10.0), sample_rate=SR)
    tr.set_loop(Loop(*SECTION, "Verse"))                           # Lead-in: Passes start at 1.0
    events = []
    for _ in range(round(5.5 * SR / 256)):                         # into Pass 2
        events += tr.render(256).events
    assert tr.pass_index == 2
    tr.set_loop(Loop(*SECTION, "Verse", count_in=TEMPO))
    b = tr.render(256)
    assert [e.kind for e in b.events] == [] and tr.pass_index == 2
    started = []
    for _ in range(round(4.0 * SR / 256)):
        started += [e for e in tr.render(256).events if e.kind == "pass_started"]
    assert started[0].pass_index == 3 and started[0].loop.count_in == TEMPO
    assert np.isclose(started[0].song_t, SECTION[0] - TEMPO.bar_s)
