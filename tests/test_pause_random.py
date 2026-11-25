"""Random user actions against the pause invariants (#50, #64), a short fixed set of seeds, one of them on a rough device
(underflows, overflows, clock drift, UI stalls). The hunt and what it checks: tests/pause_fuzz.py (run it by hand with
more seeds and steps when the pause or stream code changes). Then the hunt itself is tested: it must catch each of the
deliberately broken Transports of pause_mutants.py and pass the unbroken one on the same directed script."""
import pause_mutants
import pytest
from pause_fuzz import FAULTS, run

SEEDS = [11, 12, 13]
STEPS = 22
HOLD = ["a_start", "a_wait", "a_space", "a_hold", "a_space", "a_wait", "a_stop"]     # a 4 s pause between two waits


def test_random_actions_keep_the_song_paused_silent_and_frozen_and_one_stream_open(tmp_path):
    watched = 0
    for seed in SEEDS:
        violations, actions, n = run(seed, STEPS, home=tmp_path / str(seed))
        assert not violations, f"seed {seed}\n" + "\n".join(violations) + "\n--- actions\n" + "\n".join(actions)
        watched += n
    assert watched >= 1, f"the seeds watched only {watched} pauses: they test too little"


def test_random_actions_on_a_rough_device_keep_the_same_invariants(tmp_path):
    violations, actions, _ = run(14, STEPS, home=tmp_path, faults=FAULTS)
    assert not violations, "\n".join(violations) + "\n--- actions\n" + "\n".join(actions)


def test_the_directed_pause_passes_on_a_sound_transport(tmp_path):
    violations, actions, watched = run(1, 0, home=tmp_path, script=HOLD)
    assert not violations and watched >= 1, "\n".join(violations) + "\n" + "\n".join(actions)


@pytest.mark.parametrize("mutant", pause_mutants.NAMES)
def test_the_hunt_catches_a_deliberately_broken_transport(tmp_path, mutant):
    undo = pause_mutants.apply(mutant)
    try:
        violations, _, _ = run(1, 0, home=tmp_path, script=HOLD)
    finally:
        undo()
    assert violations, f"the hunt did not notice the {mutant} transport"
