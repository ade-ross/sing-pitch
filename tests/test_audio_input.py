import numpy as np

from sing_pitch.audio_input import ArrayInput, pick_input


def dev(index, name, ins=1):
    return {"index": index, "name": name, "max_input_channels": ins, "max_output_channels": 0}


DEVICES = [
    dev(0, "BlackHole 2ch", 2),
    dev(1, "Background Music", 2),
    dev(2, "MacBook Pro Microphone"),
    dev(3, "Yeti Stereo Microphone", 2),
    dev(4, "Speakers", 0),
]


def test_the_yeti_is_chosen_when_it_is_plugged_in():
    assert pick_input(DEVICES) == 3


def test_without_the_yeti_the_builtin_mic_is_chosen_not_a_virtual_device():
    assert pick_input([d for d in DEVICES if d["index"] != 3]) == 2


def test_a_device_without_inputs_is_never_chosen():
    assert pick_input([dev(4, "Yeti Stereo Microphone", 0), dev(2, "MacBook Pro Microphone")]) == 2


def test_nothing_recognisable_falls_back_to_the_first_real_input():
    assert pick_input([dev(7, "USB Audio Interface", 2)]) == 7
    assert pick_input([]) is None


def test_array_input_delivers_the_samples_in_chunks_without_a_mic():
    x = np.arange(1300, dtype=np.float32)
    got = []
    ai = ArrayInput(x, sample_rate=48000, chunk=512)
    ai.start(got.append)
    assert [len(c) for c in got] == [512, 512, 276]
    assert np.array_equal(np.concatenate(got), x)
