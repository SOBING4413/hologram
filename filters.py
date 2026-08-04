"""
filters.py
----------
Signal smoothing for noisy hand-landmark coordinates.

Implements:
  - LowPassFilter        : basic exponential low-pass building block
  - OneEuroFilter         : the "1 Euro Filter" (Casiez et al. 2012), which
                            adapts its cutoff frequency to the speed of
                            movement -> smooth when still, responsive when
                            moving fast. This is what removes the jittery
                            "MediaPipe shake" while keeping fast gestures
                            crisp.
  - EMAFilter             : a simpler fixed-alpha exponential moving average,
                            kept as a lightweight alternative.
  - LandmarkFilter        : applies the above independently to every
                            (x, y, z) coordinate of every landmark of every
                            tracked hand, keyed by handedness label so the
                            filter state for "Left"/"Right" persists across
                            frames.
"""

import math
import time
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# One Euro Filter
# --------------------------------------------------------------------------- #
class LowPassFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._y = None
        self._initialized = False

    def filter(self, value: float) -> float:
        if not self._initialized:
            self._y = value
            self._initialized = True
        else:
            self._y = self.alpha * value + (1.0 - self.alpha) * self._y
        return self._y

    @property
    def last_value(self):
        return self._y


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilter:
    """Classic One Euro Filter for a single scalar signal."""

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.0, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff

        self._x_filter = LowPassFilter(_smoothing_factor(1.0 / freq, mincutoff))
        self._dx_filter = LowPassFilter(_smoothing_factor(1.0 / freq, dcutoff))
        self._last_time = None
        self._last_x = None

    def filter(self, x: float, timestamp: float = None) -> float:
        if timestamp is None:
            timestamp = time.perf_counter()

        if self._last_time is not None and timestamp > self._last_time:
            t_e = timestamp - self._last_time
        else:
            t_e = 1.0 / self.freq
        self._last_time = timestamp

        # Estimate the derivative (speed of change)
        dx = 0.0 if self._last_x is None else (x - self._last_x) / max(t_e, 1e-6)
        self._dx_filter.alpha = _smoothing_factor(t_e, self.dcutoff)
        edx = self._dx_filter.filter(dx)

        # Adapt cutoff based on speed: faster movement -> higher cutoff -> less lag
        cutoff = self.mincutoff + self.beta * abs(edx)
        self._x_filter.alpha = _smoothing_factor(t_e, cutoff)
        filtered = self._x_filter.filter(x)

        self._last_x = filtered
        return filtered


# --------------------------------------------------------------------------- #
# Simple EMA (alternative / fallback smoothing)
# --------------------------------------------------------------------------- #
class EMAFilter:
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._y = None

    def filter(self, value: float) -> float:
        if self._y is None:
            self._y = value
        else:
            self._y = self.alpha * value + (1.0 - self.alpha) * self._y
        return self._y


# --------------------------------------------------------------------------- #
# Per-landmark filter manager
# --------------------------------------------------------------------------- #
class LandmarkFilter:
    """
    Applies a OneEuroFilter (or EMA fallback) independently to the x, y and z
    coordinate of every landmark, keyed by (hand_label, landmark_index, axis)
    so filter state persists correctly across frames as long as the same
    hand keeps the same handedness label.
    """

    def __init__(self, use_one_euro: bool = True, freq: float = 30.0,
                 mincutoff: float = 1.2, beta: float = 0.35, dcutoff: float = 1.0,
                 ema_alpha: float = 0.35):
        self.use_one_euro = use_one_euro
        self._params = dict(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
        self._ema_alpha = ema_alpha
        self._filters: Dict[Tuple[str, int, str], object] = {}
        self._seen_this_frame = set()

    def _get_filter(self, key):
        f = self._filters.get(key)
        if f is None:
            f = OneEuroFilter(**self._params) if self.use_one_euro else EMAFilter(self._ema_alpha)
            self._filters[key] = f
        return f

    def smooth_hand(self, hand_label: str, landmarks: List[Tuple[float, float, float]],
                     timestamp: float = None) -> List[Tuple[float, float, float]]:
        smoothed = []
        for i, (x, y, z) in enumerate(landmarks):
            if self.use_one_euro:
                fx = self._get_filter((hand_label, i, "x")).filter(x, timestamp)
                fy = self._get_filter((hand_label, i, "y")).filter(y, timestamp)
                fz = self._get_filter((hand_label, i, "z")).filter(z, timestamp)
            else:
                fx = self._get_filter((hand_label, i, "x")).filter(x)
                fy = self._get_filter((hand_label, i, "y")).filter(y)
                fz = self._get_filter((hand_label, i, "z")).filter(z)
            smoothed.append((fx, fy, fz))
            self._seen_this_frame.add((hand_label, i))
        return smoothed

    def reset_hand(self, hand_label: str):
        """Drop filter state for a hand that has disappeared from the frame."""
        keys = [k for k in self._filters if k[0] == hand_label]
        for k in keys:
            del self._filters[k]

    def prune_missing(self, active_labels: List[str]):
        """Remove filter state for any hand label not present this frame."""
        stale = {k for k in self._filters if k[0] not in active_labels}
        for k in stale:
            del self._filters[k]
