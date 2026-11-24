"""Fakes come only from flags: without them the app launches as before."""
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from sing_pitch.testmode import ECHO, parse_flags


def test_no_flags_means_no_fakes():
    flags, rest = parse_flags(["--smoke"])
    assert not flags.any and rest == ["--smoke"]


def test_the_test_mode_flags_are_read_and_the_rest_left_alone():
    flags, rest = parse_flags(["--fake-mic", "v.wav", "--fake-mic-offset", "30", "--fake-latency", "0.12",
                               "--fake-output", "--record-output", "o.wav", "--smoke"])
    assert (flags.fake_mic, flags.fake_mic_offset, flags.fake_latency) == ("v.wav", 30.0, 0.12)
    assert flags.fake_output and flags.record_output == "o.wav" and rest == ["--smoke"]
    assert parse_flags(["--fake-mic", ECHO])[0].fake_mic == ECHO


def test_record_output_needs_the_fake_output():
    with pytest.raises(SystemExit):
        parse_flags(["--record-output", "o.wav"])


def test_smoke_in_test_mode_runs_on_the_fakes_and_writes_the_recording(tmp_path):
    wav = tmp_path / "voice.wav"
    sf.write(str(wav), np.zeros(4800, np.float32), 48000)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "SING_PITCH_HOME": str(tmp_path / "home")}
    done = subprocess.run([sys.executable, "-m", "sing_pitch", "--fake-mic", str(wav), "--fake-output",
                           "--record-output", str(tmp_path / "out.wav"), "--smoke"],
                          env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert done.returncode == 0 and "smoke ok" in done.stdout, done.stderr
    assert "test mode" in done.stdout
    assert sf.info(str(tmp_path / "out.wav")).samplerate == 48000
