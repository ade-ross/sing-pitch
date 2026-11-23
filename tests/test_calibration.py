"""Latency calibration without a mic: a click track, a synthetic 'clap/sung ta' recording delayed by a known
latency, and the measurement that has to find that latency again."""
import numpy as np
import pytest
from audio import SR, harmonic, noise

from sing_pitch.calibration import click_track, measure_latency
from sing_pitch.settings import DEFAULT_LATENCY_S, Settings


def performance(click_times, delay_s, seconds, jitter_s=0.0, seed=3):
    """The user's answers to the clicks: a short sung 'ta' (harmonic burst) delay_s after each click, over
    room noise. jitter_s = how unevenly a human hits the beat."""
    rng = np.random.default_rng(seed)
    out = noise(seconds, rms=0.002)
    burst = harmonic(196.0, 0.15, amp=0.4) * np.exp(-np.arange(round(0.15 * SR)) / (0.05 * SR)).astype(np.float32)
    for c in click_times:
        i = round((c + delay_s + rng.uniform(-jitter_s, jitter_s)) * SR)
        out[i : i + len(burst)] += burst[: len(out) - i]
    return out


def test_click_track_has_evenly_spaced_audible_clicks_and_silence_between():
    audio, times = click_track(SR)
    assert len(times) >= 6 and np.allclose(np.diff(times), np.diff(times)[0])
    i = round(times[0] * SR)
    assert np.abs(audio[i : i + 200]).max() > 0.3
    assert np.abs(audio[i + round(0.2 * SR) : i + round(0.3 * SR)]).max() == 0.0
    assert audio.ndim == 2 and audio.shape[1] == 2


@pytest.mark.parametrize("delay", [0.06, 0.19, 0.31])
def test_measurement_finds_the_delay_of_answers_to_the_clicks(delay):
    audio, times = click_track(SR)
    result = measure_latency(performance(times, delay, len(audio) / SR + 0.5), SR, times)
    assert abs(result.latency_s - delay) < 0.005
    assert result.matched == len(times)


def test_an_uneven_human_still_gives_the_median_delay():
    audio, times = click_track(SR)
    result = measure_latency(performance(times, 0.2, len(audio) / SR + 0.5, jitter_s=0.03), SR, times)
    assert abs(result.latency_s - 0.2) < 0.02
    assert result.spread_s < 0.05


def test_silence_gives_no_result_instead_of_a_number():
    audio, times = click_track(SR)
    result = measure_latency(noise(len(audio) / SR + 0.5, rms=0.002), SR, times)
    assert result.latency_s is None and result.matched == 0


def test_calibrated_latency_is_stored_per_output_device_and_becomes_the_default(tmp_path):
    s = Settings.load(tmp_path)
    assert s.latency_for("WF-1000XM5") == DEFAULT_LATENCY_S and not s.is_calibrated("WF-1000XM5")
    s.set_latency("WF-1000XM5", 0.31)
    s.set_latency("Scarlett 2i2 USB", 0.07)
    s.save()
    again = Settings.load(tmp_path)
    assert again.latency_for("WF-1000XM5") == 0.31 and again.is_calibrated("WF-1000XM5")
    assert again.latency_for("Scarlett 2i2 USB") == 0.07
    assert again.latency_for("Some new device") == 0.07          # the last calibration is the best guess


def test_headphones_confirmation_is_remembered_per_device(tmp_path):
    s = Settings.load(tmp_path)
    s.confirm_headphones("Scarlett 2i2 USB")
    s.save()
    assert Settings.load(tmp_path).headphones_confirmed("Scarlett 2i2 USB")
    assert not Settings.load(tmp_path).headphones_confirmed("LG HDR 4K")


def test_a_corrupt_settings_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "settings.json").write_text("{not json")
    assert Settings.load(tmp_path).latency_for("x") == DEFAULT_LATENCY_S


def test_calibration_run_uses_the_same_clocks_as_a_melody_take():
    from sing_pitch.audio_input import ArrayInput
    from sing_pitch.calibration import CalibrationRun

    class FakePlayer:
        def __init__(self, audio, rate):
            self.audio, self.started, self.stopped = audio, False, False

        def start(self):
            self.started = True

        def position_s(self):
            return 0.05                         # the clicks were already 50 ms in when the mic started

        def stop(self):
            self.stopped = True

    _, times = click_track(SR)
    # answers 0.24 s after each click as heard, on the input clock that started 0.05 s into the click track
    mic = performance(times - 0.05, 0.24, times[-1] + 1.5)
    run = CalibrationRun(ArrayInput(mic, SR, 256), player_factory=FakePlayer)
    run.start()
    result = run.finish()
    assert abs(result.latency_s - 0.24) < 0.008


def test_the_calibrate_command_measures_and_stores_the_latency_for_the_output(tmp_path, capsys):
    from sing_pitch.audio_input import ArrayInput
    from sing_pitch.calibration import CalibrationRun
    from sing_pitch.cli import calibrate_main

    class Player:
        def __init__(self, audio, rate):
            pass

        def start(self):
            pass

        def position_s(self):
            return 256 / SR

        def stop(self):
            pass

    _, times = click_track(SR)
    mic = performance(times, 0.15, times[-1] + 1.5)
    make_run = lambda: CalibrationRun(ArrayInput(mic, SR, 256), player_factory=Player)   # noqa: E731
    assert calibrate_main([], make_run=make_run, output_name="WF-1000XM5", home=tmp_path, wait=lambda s: None) == 0
    assert abs(Settings.load(tmp_path).latency_for("WF-1000XM5") - 0.15) < 0.008
    assert "WF-1000XM5" in capsys.readouterr().out
