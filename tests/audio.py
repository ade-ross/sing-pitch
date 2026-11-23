"""Synthesised audio. Voice-like HARMONIC tones: SwiftF0 gives pure sines low confidence and up to 9 cents bias."""
import numpy as np

SR = 48000


def harmonic(hz: float, seconds: float, amp: float = 0.3, sr: int = SR, harmonics: int = 20) -> np.ndarray:
    t = np.arange(round(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, harmonics + 1) if hz * k < sr / 2)
    return (amp * x / np.max(np.abs(x))).astype(np.float32)


def quiet(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(round(seconds * sr), dtype=np.float32)


def noise(seconds: float, rms: float = 0.003, sr: int = SR, seed: int = 1) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(round(seconds * sr)) * rms).astype(np.float32)
