"""
renderer.py
-----------
Modern OpenGL (moderngl / GLSL) renderer for the hand-mesh hologram effect.

Render order every frame:
    1. Webcam frame as a full-screen tinted/darkened background quad.
    2. Delaunay wireframe mesh connecting both hands (thin white glow,
       ~20% alpha), additive blending.
    3. Per-hand skeleton bones (thicker glow, colored per hand), additive
       blending.
    4. Landmark points (glowing dots), additive blending.
    5. HUD overlay (FPS + per-point coordinates), pre-rendered to a CPU
       RGBA canvas via OpenCV and uploaded as one texture, standard alpha
       blending.

Line and point geometry is rebuilt on the CPU every frame (cheap: at most a
couple hundred vertices for two 21-point hands) and uploaded as fresh
moderngl buffers -- simple and fast enough to comfortably hit 30-60 FPS,
at the cost of a little extra CPU->GPU traffic compared to persistent
pre-allocated buffers.
"""

from typing import Dict, List, Tuple

import cv2
import moderngl
import numpy as np

import shader
from utils import Config, ortho_matrix


class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int, config: Config):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.cfg = config

        self.ctx.enable(moderngl.BLEND)

        self._proj = ortho_matrix(0, width, height, 0)  # y-down pixel space -> NDC

        # ---- Programs ---------------------------------------------------- #
        self.bg_prog = self.ctx.program(vertex_shader=shader.BACKGROUND_VERTEX,
                                         fragment_shader=shader.BACKGROUND_FRAGMENT)
        self.line_prog = self.ctx.program(vertex_shader=shader.LINE_VERTEX,
                                           fragment_shader=shader.LINE_FRAGMENT)
        self.point_prog = self.ctx.program(vertex_shader=shader.POINT_VERTEX,
                                            fragment_shader=shader.POINT_FRAGMENT)
        self.overlay_prog = self.ctx.program(vertex_shader=shader.OVERLAY_VERTEX,
                                              fragment_shader=shader.OVERLAY_FRAGMENT)

        self.line_prog["u_proj"].write(self._proj.tobytes())
        self.point_prog["u_proj"].write(self._proj.tobytes())

        # ---- Full-screen quad (background + overlay share this) ---------- #
        quad = np.array([
            # pos.x, pos.y,   uv.x, uv.y
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype="f4")
        self._quad_vbo = self.ctx.buffer(quad.tobytes())
        self._bg_vao = self.ctx.vertex_array(
            self.bg_prog, [(self._quad_vbo, "2f 2f", "in_pos", "in_uv")])
        self._overlay_vao = self.ctx.vertex_array(
            self.overlay_prog, [(self._quad_vbo, "2f 2f", "in_pos", "in_uv")])

        # ---- Textures ------------------------------------------------------ #
        self.cam_tex = self.ctx.texture((width, height), 3)
        self.cam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.overlay_tex = self.ctx.texture((width, height), 4)
        self.overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self._overlay_canvas = np.zeros((height, width, 4), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def resize(self, width: int, height: int):
        self.width, self.height = width, height
        self._proj = ortho_matrix(0, width, height, 0)
        self.line_prog["u_proj"].write(self._proj.tobytes())
        self.point_prog["u_proj"].write(self._proj.tobytes())

        self.cam_tex.release()
        self.cam_tex = self.ctx.texture((width, height), 3)
        self.cam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.overlay_tex.release()
        self.overlay_tex = self.ctx.texture((width, height), 4)
        self.overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self._overlay_canvas = np.zeros((height, width, 4), dtype=np.uint8)

    def render(self, frame_rgb: np.ndarray, mesh: Dict, hands: List[Dict], fps: float):
        w, h = self.width, self.height
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self._draw_background(frame_rgb)

        # Additive blending for the glowing hologram elements
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

        points_px = self._points_to_pixels(mesh["points"])

        if self.cfg.mesh_alpha > 0.0 and len(mesh["edges"]) > 0:
            self._draw_lines(points_px, mesh["edges"],
                              color=self.cfg.mesh_color,
                              alpha=self.cfg.mesh_alpha,
                              width_px=self.cfg.mesh_line_width_px,
                              glow_power=self.cfg.mesh_glow_power)

        if len(mesh["skeleton"]) > 0:
            self._draw_skeleton(points_px, mesh["skeleton"], mesh["hand_ranges"])

        if len(points_px) > 0:
            self._draw_points(points_px, mesh["hand_ranges"])

        # Standard alpha blending for the HUD text
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        if self.cfg.show_hud:
            self._draw_hud(points_px, mesh["hand_ranges"], hands, fps)

    def release(self):
        for obj in (self._quad_vbo, self.cam_tex, self.overlay_tex,
                    self.bg_prog, self.line_prog, self.point_prog, self.overlay_prog,
                    self._bg_vao, self._overlay_vao):
            try:
                obj.release()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Internal drawing helpers
    # ------------------------------------------------------------------ #
    def _draw_background(self, frame_rgb: np.ndarray):
        # OpenGL textures are bottom-left origin; flip vertically once here.
        flipped = np.flipud(np.ascontiguousarray(frame_rgb))
        self.cam_tex.write(flipped.tobytes())
        self.cam_tex.use(location=0)
        self.bg_prog["u_tex"].value = 0
        self.bg_prog["u_brightness"].value = self.cfg.background_brightness
        self.bg_prog["u_tint"].value = self.cfg.hologram_tint
        self._bg_vao.render(moderngl.TRIANGLES)

    def _points_to_pixels(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        px = np.empty((len(points), 2), dtype=np.float32)
        px[:, 0] = points[:, 0] * self.width
        px[:, 1] = points[:, 1] * self.height
        return px

    def _build_line_geometry(self, points_px: np.ndarray, edges: List[Tuple[int, int]],
                              color: Tuple[float, float, float], alpha: float,
                              width_px: float) -> np.ndarray:
        """Each edge -> a quad (2 triangles, 6 vertices) in pixel space.
        Vertex layout: pos.x, pos.y, side(-1..1), color.r,g,b, alpha
        """
        if not edges:
            return np.zeros((0, 7), dtype="f4")

        half_w = width_px * 0.5
        verts = np.empty((len(edges) * 6, 7), dtype="f4")
        r, g, b = color

        for k, (i, j) in enumerate(edges):
            p0 = points_px[i]
            p1 = points_px[j]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = (dx * dx + dy * dy) ** 0.5
            if length < 1e-6:
                nx, ny = 0.0, 0.0
            else:
                nx, ny = -dy / length * half_w, dx / length * half_w

            a = (p0[0] + nx, p0[1] + ny)  # side +1
            b_ = (p0[0] - nx, p0[1] - ny)  # side -1
            c = (p1[0] + nx, p1[1] + ny)  # side +1
            d = (p1[0] - nx, p1[1] - ny)  # side -1

            base = k * 6
            verts[base + 0] = (*a, 1.0, r, g, b, alpha)
            verts[base + 1] = (*b_, -1.0, r, g, b, alpha)
            verts[base + 2] = (*c, 1.0, r, g, b, alpha)
            verts[base + 3] = (*c, 1.0, r, g, b, alpha)
            verts[base + 4] = (*b_, -1.0, r, g, b, alpha)
            verts[base + 5] = (*d, -1.0, r, g, b, alpha)

        return verts

    def _draw_line_batch(self, verts: np.ndarray, glow_power: float):
        if len(verts) == 0:
            return
        vbo = self.ctx.buffer(verts.tobytes())
        vao = self.ctx.vertex_array(
            self.line_prog, [(vbo, "2f 1f 3f 1f", "in_pos", "in_side", "in_color", "in_alpha")])
        self.line_prog["u_glow_power"].value = glow_power
        vao.render(moderngl.TRIANGLES)
        vao.release()
        vbo.release()

    def _draw_lines(self, points_px, edges, color, alpha, width_px, glow_power):
        verts = self._build_line_geometry(points_px, edges, color, alpha, width_px)
        self._draw_line_batch(verts, glow_power)

    def _draw_skeleton(self, points_px, skeleton_edges, hand_ranges):
        # Color each bone according to which hand it belongs to.
        for label, start, end in hand_ranges:
            color = (self.cfg.left_hand_color if label == "Left" else self.cfg.right_hand_color)
            this_hand_edges = [(i, j) for (i, j) in skeleton_edges if start <= i < end and start <= j < end]
            if not this_hand_edges:
                continue
            verts = self._build_line_geometry(
                points_px, this_hand_edges, color,
                self.cfg.skeleton_alpha, self.cfg.skeleton_line_width_px)
            self._draw_line_batch(verts, self.cfg.skeleton_glow_power)

    def _build_point_geometry(self, points_px: np.ndarray, indices: List[int],
                               color: Tuple[float, float, float], alpha: float,
                               radius_px: float) -> np.ndarray:
        """Each point -> a quad billboard. Layout: pos.x,pos.y, uv.x,uv.y, color, alpha"""
        r, g, b = color
        verts = np.empty((len(indices) * 6, 8), dtype="f4")
        for k, idx in enumerate(indices):
            cx, cy = points_px[idx]
            x0, x1 = cx - radius_px, cx + radius_px
            y0, y1 = cy - radius_px, cy + radius_px
            base = k * 6
            verts[base + 0] = (x0, y0, -1.0, -1.0, r, g, b, alpha)
            verts[base + 1] = (x1, y0, 1.0, -1.0, r, g, b, alpha)
            verts[base + 2] = (x0, y1, -1.0, 1.0, r, g, b, alpha)
            verts[base + 3] = (x0, y1, -1.0, 1.0, r, g, b, alpha)
            verts[base + 4] = (x1, y0, 1.0, -1.0, r, g, b, alpha)
            verts[base + 5] = (x1, y1, 1.0, 1.0, r, g, b, alpha)
        return verts

    def _draw_points(self, points_px, hand_ranges):
        for label, start, end in hand_ranges:
            color = (self.cfg.left_hand_color if label == "Left" else self.cfg.right_hand_color)
            indices = list(range(start, end))
            if not indices:
                continue
            verts = self._build_point_geometry(
                points_px, indices, color, self.cfg.point_alpha, self.cfg.point_radius_px)
            vbo = self.ctx.buffer(verts.tobytes())
            vao = self.ctx.vertex_array(
                self.point_prog, [(vbo, "2f 2f 3f 1f", "in_pos", "in_uv", "in_color", "in_alpha")])
            self.point_prog["u_glow_power"].value = self.cfg.point_glow_power
            vao.render(moderngl.TRIANGLES)
            vao.release()
            vbo.release()

    # ------------------------------------------------------------------ #
    # HUD (text) overlay
    # ------------------------------------------------------------------ #
    def _draw_hud(self, points_px, hand_ranges, hands, fps):
        canvas = self._overlay_canvas
        canvas[:] = 0

        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255, 255)
        dim = (255, 255, 255, 160)

        # FPS + general info, top-left
        cv2.putText(canvas, f"FPS: {fps:5.1f}", (16, 30), font, 0.7, white, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Hands detected: {len(hands)}", (16, 56), font, 0.55, dim, 1, cv2.LINE_AA)

        # Per-hand label near the wrist
        for label, start, end in hand_ranges:
            if end <= start:
                continue
            wrist = points_px[start]
            cv2.putText(canvas, label, (int(wrist[0]) - 20, int(wrist[1]) + 34),
                        font, 0.6, white, 2, cv2.LINE_AA)

        # Per-point normalized coordinates
        if self.cfg.show_coordinates:
            for label, start, end in hand_ranges:
                for i in range(start, end):
                    x_px, y_px = points_px[i]
                    lm_idx = i - start
                    text = f"{lm_idx}:({x_px/self.width:.2f},{y_px/self.height:.2f})"
                    cv2.putText(canvas, text, (int(x_px) + 6, int(y_px) - 6),
                                font, self.cfg.hud_font_scale, dim,
                                self.cfg.hud_thickness, cv2.LINE_AA)

        flipped = np.flipud(canvas)
        self.overlay_tex.write(np.ascontiguousarray(flipped).tobytes())
        self.overlay_tex.use(location=0)
        self.overlay_prog["u_tex"].value = 0
        self._overlay_vao.render(moderngl.TRIANGLES)
