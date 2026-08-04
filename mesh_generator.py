"""
mesh_generator.py
------------------
Turns the raw (smoothed) landmark point clouds from one or two hands into
renderable geometry:

    - "points"    : Nx3 float32 array of all landmark coordinates
                     (normalized x, y in [0,1], z relative depth), hands
                     concatenated in detection order.
    - "triangles" : Mx3 int32 array of triangle indices from a 2D Delaunay
                     triangulation over ALL points from BOTH hands at once.
                     This is what produces the "web" that dynamically
                     stretches and connects the two hands together.
    - "edges"     : deduplicated (i, j) index pairs extracted from the
                     triangles, used to draw the mesh as a WIREFRAME instead
                     of filled triangles.
    - "skeleton"  : per-hand bone connections (MediaPipe HAND_CONNECTIONS),
                     offset to index into the same concatenated point array.
    - "hand_ranges": list of (label, start_index, end_index) so the renderer
                     can color/label each hand's points separately.

Delaunay triangulation is computed with scipy.spatial.Delaunay. If triangulation
fails (e.g. degenerate/collinear points, fewer than 3 points), we simply
return an empty triangle set and let the renderer fall back to skeleton-only.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import QhullError


class MeshGenerator:
    def __init__(self, hand_connections: List[Tuple[int, int]], max_edge_length: float = 0.38):
        self.hand_connections = hand_connections
        self.max_edge_length = max_edge_length

    def build(self, hands: List[Dict]) -> Dict:
        all_points: List[Tuple[float, float, float]] = []
        hand_ranges = []
        skeleton: List[Tuple[int, int]] = []

        offset = 0
        for hand in hands:
            lms = hand["landmarks"]
            start = offset
            all_points.extend(lms)
            offset += len(lms)
            hand_ranges.append((hand["label"], start, offset))

            for a, b in self.hand_connections:
                skeleton.append((a + start, b + start))

        points = np.asarray(all_points, dtype=np.float32) if all_points else np.zeros((0, 3), dtype=np.float32)

        triangles = np.zeros((0, 3), dtype=np.int32)
        edges: List[Tuple[int, int]] = []

        if len(points) >= 3:
            pts2d = points[:, :2].astype(np.float64)
            # Guard against fully-degenerate (collinear) point sets, which
            # make qhull raise instead of returning gracefully.
            try:
                tri = Delaunay(pts2d, qhull_options="QJ")  # QJ jitters to avoid precision issues
                triangles = tri.simplices.astype(np.int32)
                edges = self._unique_edges(triangles)
                edges = self._prune_long_edges(points, edges)
            except (QhullError, Exception):
                triangles = np.zeros((0, 3), dtype=np.int32)
                edges = []

        return {
            "points": points,
            "triangles": triangles,
            "edges": edges,
            "skeleton": skeleton,
            "hand_ranges": hand_ranges,
        }

    def _prune_long_edges(self, points: np.ndarray, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if self.max_edge_length <= 0:
            return edges
        kept = []
        max_len_sq = self.max_edge_length * self.max_edge_length
        for i, j in edges:
            dx = float(points[i, 0] - points[j, 0])
            dy = float(points[i, 1] - points[j, 1])
            if dx * dx + dy * dy <= max_len_sq:
                kept.append((i, j))
        return kept

    @staticmethod
    def _unique_edges(triangles: np.ndarray) -> List[Tuple[int, int]]:
        edge_set = set()
        for tri in triangles:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            for a, b in ((i0, i1), (i1, i2), (i2, i0)):
                edge_set.add((a, b) if a < b else (b, a))
        return list(edge_set)
