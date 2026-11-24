"""Stretched stems are cached per song x Speed inside the song's folder; a cached Speed is not recomputed."""
import numpy as np
import soundfile as sf
from melody_helpers import library_song, reference_line

from sing_pitch.playback.stretch import SpeedCache

SR = 48000


class CountingStretcher:
    """Stands in for Rubber Band: repeats frames so the length is original / speed, and counts its calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, audio, sample_rate, speed):
        self.calls.append((len(audio), speed))
        n = round(len(audio) / speed)
        return np.ascontiguousarray(audio[np.minimum((np.arange(n) * speed).astype(int), len(audio) - 1)])


def song(tmp_path):
    return library_song(tmp_path, reference_line([(0, 1.0)]), seconds=1.0, sr=SR)


def test_the_first_use_renders_both_stems_the_second_does_not(tmp_path):
    s = song(tmp_path)
    stretcher = CountingStretcher()
    cache = SpeedCache(s.dir, SR, stretcher)
    assert not cache.has(70)
    backing, vocal = cache.get(70)
    assert cache.has(70) and len(stretcher.calls) == 2                      # Backing and vocal
    assert backing.shape == (round(SR / 0.7), 2) and vocal.shape == backing.shape
    cache.get(70)
    SpeedCache(s.dir, SR, stretcher).get(70)                                # a new cache object, same folder
    assert len(stretcher.calls) == 2
    assert (s.dir / "speed" / "70" / "backing.wav").exists()


def test_each_speed_is_cached_on_its_own_and_100_percent_is_the_original(tmp_path):
    s = song(tmp_path)
    stretcher = CountingStretcher()
    cache = SpeedCache(s.dir, SR, stretcher)
    cache.get(70), cache.get(50)
    assert len(stretcher.calls) == 4
    backing, _ = cache.get(100)
    assert len(stretcher.calls) == 4 and len(backing) == SR and cache.has(100)


def test_a_song_without_a_vocal_stem_stretches_only_the_backing(tmp_path):
    s = song(tmp_path)
    s.vocals_path.unlink()
    stretcher = CountingStretcher()
    backing, vocal = SpeedCache(s.dir, SR, stretcher).get(80)
    assert vocal is None and len(stretcher.calls) == 1 and len(backing) == round(SR / 0.8)


def test_a_failed_render_leaves_no_half_cached_speed(tmp_path):
    s = song(tmp_path)

    def broken(audio, sample_rate, speed):
        raise RuntimeError("out of memory")

    cache = SpeedCache(s.dir, SR, broken)
    try:
        cache.get(70)
    except RuntimeError:
        pass
    assert not cache.has(70)
    assert cache.get(100)[0] is not None
