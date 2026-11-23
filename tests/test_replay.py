from datetime import datetime

from frames import hz_at, silence, sung

from sing_pitch.reference import Band, NearestNote
from sing_pitch.replay import Replay
from sing_pitch.take_engine import TakeEngine, replay_scores
from sing_pitch.take_store import load_frames, save_take


def saved_take(tmp_path):
    engine = TakeEngine(NearestNote(), started_at=datetime(2026, 11, 23, 21, 5, 9))
    engine.push(sung(hz_at(30), 1.0) + silence(0.5, start=1.0) + sung(hz_at(5), 1.0, start=1.5))
    return save_take(tmp_path, engine.stop())


def test_replaying_a_saved_line_gives_one_score_per_frame_with_the_bands_it_had():
    frames = sung(hz_at(30), 1.0) + silence(0.5, start=1.0) + sung(hz_at(5), 1.0, start=1.5)
    scores = replay_scores(frames, NearestNote())
    assert len(scores) == len(frames)
    assert scores[30].band is Band.YELLOW and scores[-1].band is Band.GREEN
    assert all(s.band is None for s in scores if 1.05 < s.t < 1.45)


def test_replay_reveals_frames_as_the_clock_passes_them(tmp_path):
    record = saved_take(tmp_path)
    replay = Replay(replay_scores(load_frames(tmp_path, record.id), NearestNote()))
    first = replay.advance(0.5)
    assert first and first[-1].t <= 0.5 and not replay.finished
    rest = replay.advance(100.0)
    assert replay.finished
    assert len(first) + len(rest) == len(record.frames)
    assert replay.advance(101.0) == []
