"""Random user actions against the pause invariants (#50), a short fixed set of seeds. The hunt itself and what it
checks: tests/pause_fuzz.py (run it by hand with more seeds and steps when the pause or stream code changes)."""
import pytest
from pause_fuzz import run

SEEDS = [11, 12, 13]
STEPS = 22


def test_random_actions_keep_the_song_paused_silent_and_frozen_and_one_stream_open(tmp_path):
    watched = 0
    for seed in SEEDS:
        violations, actions, n = run(seed, STEPS, home=tmp_path / str(seed))
        assert not violations, f"seed {seed}\n" + "\n".join(violations) + "\n--- actions\n" + "\n".join(actions)
        watched += n
    assert watched >= 3, f"the seeds watched only {watched} pauses: they test too little"
