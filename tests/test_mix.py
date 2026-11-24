"""The pure mixing step of the Backing player: output = backing*g_b + vocal*g_v, gains ramped across the block."""
import numpy as np

from sing_pitch.playback.mix import mix_block


def stereo(value, frames):
    return np.full((frames, 2), value, np.float32)


def test_constant_gains_mix_linearly():
    out = mix_block(stereo(0.5, 8), stereo(0.25, 8), (1.0, 1.0), (1.0, 1.0))
    assert np.allclose(out, 0.5 * 1.0 + 0.25 * 1.0)
    out = mix_block(stereo(0.5, 8), stereo(0.25, 8), (0.5, 2.0), (0.5, 2.0))
    assert np.allclose(out, 0.5 * 0.5 + 0.25 * 2.0)


def test_vocal_gain_zero_silences_the_vocal():
    out = mix_block(stereo(0.5, 8), stereo(0.9, 8), (1.0, 0.0), (1.0, 0.0))
    assert np.allclose(out, 0.5)


def test_a_missing_vocal_leaves_the_backing_alone():
    out = mix_block(stereo(0.5, 8), None, (1.0, 0.0), (1.0, 1.0))
    assert np.allclose(out, 0.5)


def test_gains_ramp_linearly_across_the_block_without_a_step():
    frames = 100
    out = mix_block(stereo(0.0, frames), stereo(1.0, frames), (1.0, 0.0), (1.0, 1.0))[:, 0]
    assert out[0] < 0.02 and abs(out[-1] - 1.0) < 1e-6          # from the old gain to the new one
    steps = np.diff(out)
    assert np.allclose(steps, steps[0]) and steps[0] < 0.011     # even slope: the biggest jump is 1/frames


def test_the_next_block_starts_where_the_ramp_ended():
    first = mix_block(stereo(0.0, 64), stereo(1.0, 64), (1.0, 0.0), (1.0, 1.0))[:, 0]
    second = mix_block(stereo(0.0, 64), stereo(1.0, 64), (1.0, 1.0), (1.0, 1.0))[:, 0]
    assert abs(second[0] - first[-1]) < 1.0 / 64 + 1e-6


def test_a_short_chunk_at_the_song_end_is_padded_with_silence():
    out = mix_block(stereo(1.0, 3), stereo(1.0, 2), (1.0, 1.0), (1.0, 1.0), frames=5)
    assert out.shape == (5, 2)
    assert np.allclose(out[:, 0], [2.0, 2.0, 1.0, 0.0, 0.0])
