"""The whole non-UI jump path with the REAL Backing player and Transport on a fake output stream: player.seek() ->
audio callback -> "jumped" event -> runner -> Take boundary, with the output and a fake mic advanced in step."""
import csv

import numpy as np
from melody_helpers import reference_line
from test_backing import FakeStream
from test_take_boundary_on_jump import LATENCY, MELODY, MIC_SR, Mic, VoiceTracker, voice

from sing_pitch.live import LiveRunner
from sing_pitch.playback import BackingPlayer
from sing_pitch.reference import Melody

OUT_SR, BLOCK, MIC_BLOCK = 48000, 960, 320       # 20 ms of output and of mic per step


def test_a_seek_on_the_real_player_splits_the_run_into_two_in_tune_takes(tmp_path):
    player = BackingPlayer(np.zeros((round(4.6 * OUT_SR), 2), np.float32), OUT_SR,
                           stream_factory=FakeStream, sample_rate=OUT_SR)
    jump_s = 175 * BLOCK / OUT_SR                 # the seek is made before the 175th block: output time 3.5 s

    def heard(out_s):                             # the schedule alone: linear, then back to the start at the jump
        return out_s if out_s < jump_s else out_s - jump_s

    mic = Mic()
    # the output is one block ahead of the mic when the mic starts, and the first mic chunk is one block long:
    # the runner reads "output at input zero" = 0
    runner = LiveRunner(mic, VoiceTracker(voice(heard, 7.5, out_at_zero=0.0)), Melody(reference_line(MELODY), "do-re-mi"),
                        tmp_path, backing=player, latency_s=LATENCY)
    runner.start()
    stream = player._stream
    stream.pull(BLOCK)
    for pulled in range(1, 380):
        if pulled == 175:
            player.seek(0.0)
        mic.on_chunk(np.zeros(MIC_BLOCK, np.float32))
        stream.pull(BLOCK)
    record = runner.stop()
    rows = list(csv.DictReader(open(tmp_path / "takes.csv")))
    assert len(rows) == 2 and record.id == rows[1]["id"]
    assert all(float(r["pct_in_tune"]) >= 98.0 and float(r["duration_s"]) > 3.0 for r in rows)
    assert abs(float(rows[0]["duration_s"]) - (jump_s + LATENCY)) < 0.1      # cut where the voice answers the jump
