"""
utils.py
--------
Helper utilities shared across the project: FPS counter, coordinate
conversion helpers, small math functions (orthographic projection matrix),
and a single Config dataclass holding every tunable visual parameter so the
whole "look" of the hologram can be tweaked from one place.
"""

import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# FPS counter
# --------------------------------------------------------------------------- #
class FPSCounter:
    """Exponentially-smoothed FPS counter (avoids a jittery on-screen number)."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            instant = 1.0 / dt
            self.fps = (
                instant if self.fps <= 0 else
                self.smoothing * self.fps + (1.0 - self.smoothing) * instant
            )
        return self.fps


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def ortho_matrix(left: float, right: float, bottom: float, top: float,
                  near: float = -1.0, far: float = 1.0) -> np.ndarray:
    """
    Build a standard orthographic projection matrix that maps
    [left, right] x [bottom, top] (pixel space, y-down) to OpenGL NDC
    [-1, 1] x [-1, 1] (y-up). Returned as a flat column-major float32 array
    ready to be uploaded to a `mat4` uniform.
    """
    rl = right - left
    tb = top - bottom
    fn = far - near

    m = np.identity(4, dtype="f4")
    m[0, 0] = 2.0 / rl
    m[1, 1] = 2.0 / tb
    m[2, 2] = -2.0 / fn
    m[0, 3] = -(right + left) / rl
    m[1, 3] = -(top + bottom) / tb
    m[2, 3] = -(far + near) / fn
    # moderngl expects column-major (Fortran order) flattening
    return m.T.astype("f4").copy()


def normalized_to_pixel(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """Convert a MediaPipe normalized landmark coordinate to pixel space."""
    return x * width, y * height


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------- #
# Global visual configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # Window / capture
    window_title: str = "Hand Mesh Hologram"
    capture_width: int = 1280
    capture_height: int = 720
    max_hands: int = 2
    mirror: bool = True

    # Detection
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.6

    # Smoothing (One Euro Filter)
    smoothing_enabled: bool = True
    one_euro_freq: float = 30.0
    one_euro_mincutoff: float = 1.2
    one_euro_beta: float = 0.35
    one_euro_dcutoff: float = 1.0

    # Background
    background_brightness: float = 0.55
    hologram_tint: Tuple[float, float, float] = (0.55, 1.0, 1.0)  # cyan-ish

    # Mesh (Delaunay wireframe connecting both hands)
    mesh_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # white
    mesh_alpha: float = 0.20
    mesh_line_width_px: float = 1.4
    mesh_glow_power: float = 3.0

    # Skeleton (per-hand bones)
    skeleton_alpha: float = 0.55
    skeleton_line_width_px: float = 3.2
    skeleton_glow_power: float = 2.2
    left_hand_color: Tuple[float, float, float] = (0.25, 0.9, 1.0)   # cyan
    right_hand_color: Tuple[float, float, float] = (1.0, 0.35, 0.85)  # magenta

    # Landmarks (points)
    point_radius_px: float = 6.5
    point_alpha: float = 0.95
    point_glow_power: float = 2.0

    # HUD / text overlay
    show_hud: bool = True
    show_coordinates: bool = True
    hud_font_scale: float = 0.38
    hud_thickness: int = 1

    # Performance
    target_fps: int = 60
