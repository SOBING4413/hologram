"""
main.py
-------
Entry point for the hand-mesh hologram application.

Pipeline per frame:
    webcam frame (BGR)
        -> mirror + convert to RGB
        -> HandTracker.process()            (MediaPipe, up to 2 hands x 21 landmarks)
        -> LandmarkFilter.smooth_hand()      (One Euro Filter, per hand per axis)
        -> MeshGenerator.build()             (Delaunay wireframe connecting both hands
                                               + per-hand skeleton)
        -> Renderer.render()                 (moderngl / GLSL: glow, alpha, AA)
        -> glfw.swap_buffers()

Controls:
    ESC / Q   quit
    H         toggle HUD (FPS + per-point coordinates)
    C         toggle coordinate labels only (keeps FPS)
    M         mirror on/off
    F         toggle One-Euro smoothing on/off (raw vs filtered)
"""

import platform
import sys
import time

# env_check must run BEFORE importing cv2/mediapipe/moderngl/glfw modules
# that depend on them, so a broken/incompatible install produces one clear
# message instead of a deep traceback.
import env_check

try:
    env_check.run_all()
except env_check.EnvironmentIssue as e:
    print("=" * 62, file=sys.stderr)
    print("[SETUP DIPERLUKAN] Lingkungan Python belum siap:", file=sys.stderr)
    print("=" * 62, file=sys.stderr)
    print(str(e), file=sys.stderr)
    print("=" * 62, file=sys.stderr)
    sys.exit(1)

import cv2
import glfw
import moderngl
import numpy as np

from filters import LandmarkFilter
from hand_tracker import HandTracker
from mesh_generator import MeshGenerator
from renderer import Renderer
from utils import Config, FPSCounter


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.running = True

        # ---- Webcam -------------------------------------------------- #
        # CAP_DSHOW opens noticeably faster and more reliably on Windows;
        # fall back to CAP_ANY (default backend) on other platforms.
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(0, backend)
        if not self.cap.isOpened():
            # Retry once with the default backend in case CAP_DSHOW itself
            # is unavailable on this machine.
            self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Tidak bisa membuka webcam (index 0). Pastikan kamera "
                "terhubung, tidak sedang dipakai aplikasi lain, dan izin "
                "kamera untuk Python/terminal sudah diberikan."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Webcam opened but failed to deliver a frame.")
        self.width = frame.shape[1]
        self.height = frame.shape[0]

        # ---- Window / GL context -------------------------------------- #
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 4)  # MSAA anti-aliasing
        glfw.window_hint(glfw.RESIZABLE, True)

        self.window = glfw.create_window(self.width, self.height, cfg.window_title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError(
                "Failed to create a GLFW/OpenGL window. Make sure your GPU "
                "driver supports OpenGL 3.3 core profile."
            )

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync; set to 0 to uncap for benchmarking

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        self.ctx = moderngl.create_context()

        # ---- Pipeline components ---------------------------------------- #
        self.tracker = HandTracker(
            max_hands=cfg.max_hands,
            detection_confidence=cfg.detection_confidence,
            tracking_confidence=cfg.tracking_confidence,
        )
        self.mesh_gen = MeshGenerator(hand_connections=self.tracker.connections)
        self.filter = LandmarkFilter(
            use_one_euro=cfg.smoothing_enabled,
            freq=cfg.one_euro_freq,
            mincutoff=cfg.one_euro_mincutoff,
            beta=cfg.one_euro_beta,
            dcutoff=cfg.one_euro_dcutoff,
        )
        self.renderer = Renderer(self.ctx, self.width, self.height, cfg)
        self.fps_counter = FPSCounter()

    # ------------------------------------------------------------------ #
    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            self.running = False
        elif key == glfw.KEY_H:
            self.cfg.show_hud = not self.cfg.show_hud
        elif key == glfw.KEY_C:
            self.cfg.show_coordinates = not self.cfg.show_coordinates
        elif key == glfw.KEY_M:
            self.cfg.mirror = not self.cfg.mirror
        elif key == glfw.KEY_F:
            self.cfg.smoothing_enabled = not self.cfg.smoothing_enabled
            self.filter.use_one_euro = self.cfg.smoothing_enabled

    def _on_resize(self, window, width, height):
        if width == 0 or height == 0:
            return
        self.width, self.height = width, height
        self.renderer.resize(width, height)

    # ------------------------------------------------------------------ #
    def run(self):
        try:
            while self.running and not glfw.window_should_close(self.window):
                glfw.poll_events()

                ok, frame_bgr = self.cap.read()
                if not ok:
                    continue

                if self.cfg.mirror:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                if (frame_bgr.shape[1], frame_bgr.shape[0]) != (self.width, self.height):
                    frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                hands = self.tracker.process(frame_rgb)

                active_labels = [h["label"] for h in hands]
                self.filter.prune_missing(active_labels)

                timestamp = time.perf_counter()
                for hand in hands:
                    hand["landmarks"] = self.filter.smooth_hand(
                        hand["label"], hand["landmarks"], timestamp)

                mesh = self.mesh_gen.build(hands)

                fps = self.fps_counter.tick()
                self.renderer.render(frame_rgb, mesh, hands, fps)

                glfw.swap_buffers(self.window)
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            self.tracker.close()
        except Exception:
            pass
        try:
            self.renderer.release()
        except Exception:
            pass
        if self.cap is not None:
            self.cap.release()
        if self.window is not None:
            glfw.destroy_window(self.window)
        glfw.terminate()


def main():
    cfg = Config()
    try:
        app = App(cfg)
    except RuntimeError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
