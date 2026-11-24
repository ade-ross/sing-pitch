"""Ticket #42: the song's tempo for the Count-in. Detected once from the Backing (a beat tracker) and stored in the song
folder as tempo.json (bpm + beat times); computed on first use for songs imported before; "unknown" when no beat tracker
is installed. Simone's own BPM (prog songs fool detectors) lives in player.json and wins."""
import json

import numpy as np
import pytest
from importer_fakes import FakeExtractor, FakeSeparator, song_file, write_wav

from sing_pitch.library import Importer, Library
from sing_pitch.library.tempo import TEMPO, Tempo, TempoUnavailable, load_tempo, song_tempo


class FakeDetector:
    def __init__(self, tempo=Tempo(96.0, (0.3, 0.925, 1.55))):
        self.tempo, self.calls = tempo, []

    def __call__(self, backing):
        self.calls.append(backing)
        if isinstance(self.tempo, Exception):
            raise self.tempo
        return self.tempo


def test_the_first_use_detects_and_stores_the_tempo_and_later_uses_read_it(tmp_path):
    detect = FakeDetector()
    (tmp_path / "backing.wav").write_bytes(b"")
    assert song_tempo(tmp_path, detect) == Tempo(96.0, (0.3, 0.925, 1.55))
    saved = json.loads((tmp_path / TEMPO).read_text())
    assert saved["bpm"] == 96.0 and saved["beats_s"] == [0.3, 0.925, 1.55]
    assert song_tempo(tmp_path, detect) == Tempo(96.0, (0.3, 0.925, 1.55)) and len(detect.calls) == 1


def test_without_a_beat_tracker_the_tempo_is_unknown_and_nothing_is_written(tmp_path):
    detect = FakeDetector(TempoUnavailable("no librosa"))
    assert song_tempo(tmp_path, detect) is None
    assert not (tmp_path / TEMPO).exists()


def test_a_detector_that_crashes_gives_unknown_and_a_broken_file_reads_as_none(tmp_path):
    assert song_tempo(tmp_path, FakeDetector(RuntimeError("bad audio"))) is None
    (tmp_path / TEMPO).write_text("{not json")
    assert load_tempo(tmp_path) is None
    (tmp_path / TEMPO).write_text(json.dumps({"bpm": -3, "beats_s": []}))
    assert load_tempo(tmp_path) is None


def test_an_import_stores_the_tempo_of_the_backing(tmp_path):
    detect = FakeDetector()
    imp = Importer(Library(tmp_path / "home"), separator=FakeSeparator(), extractor=FakeExtractor(), tempo=detect)
    song = imp.import_file(song_file(tmp_path / "a.wav"))
    assert detect.calls == [detect.calls[0]] and detect.calls[0].name == "backing.wav"
    assert load_tempo(Library(tmp_path / "home").get(song.id).dir) == detect.tempo


def test_an_import_without_a_beat_tracker_still_imports_the_song(tmp_path):
    imp = Importer(Library(tmp_path / "home"), separator=FakeSeparator(), extractor=FakeExtractor(),
                   tempo=FakeDetector(TempoUnavailable("no librosa")))
    song = imp.import_file(song_file(tmp_path / "a.wav"))
    assert load_tempo(Library(tmp_path / "home").get(song.id).dir) is None


def test_the_real_beat_tracker_finds_the_tempo_of_a_click_track(tmp_path):
    pytest.importorskip("librosa")
    from sing_pitch.library.tempo import detect_tempo

    sr, bpm = 22050, 100.0
    audio = np.zeros(int(20 * sr), np.float32)
    t = np.arange(int(0.03 * sr)) / sr
    click = np.exp(-t / 0.005) * np.sin(2 * np.pi * 1000 * t)
    for k in range(int(20 * bpm / 60)):
        at = int(k * 60 / bpm * sr)
        audio[at: at + len(click)] += click[: len(audio) - at]
    tempo = detect_tempo(write_wav(tmp_path / "backing.wav", audio, sr))
    assert abs(tempo.bpm - bpm) < 2.0
    assert len(tempo.beats_s) > 20 and abs(np.median(np.diff(tempo.beats_s)) - 0.6) < 0.02
