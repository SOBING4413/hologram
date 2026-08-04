"""
mesh_generator.py
------------------
Turns the raw (smoothed) landmark point clouds from one or two hands into
renderable geometry:

    - "points"    : Nx3 float32 array of all landmark coordinates
                     (normalized x, y in [0,1], z relative depth), hands
                     concatenated in detection order.
    - "triangles" : Mx3 int32 array of all valid Delaunay triangle indices.
    - "panel_triangles": Kx3 int32 array containing only bridge/object-surface
                     triangles that span more than one hand. These are the
                     only triangles filled with texture, so the texture appears
                     between hands instead of covering each whole hand.
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
        panel_triangles = np.zeros((0, 3), dtype=np.int32)

        if len(points) >= 3:
            pts2d = points[:, :2].astype(np.float64)
            # Guard against fully-degenerate (collinear) point sets, which
            # make qhull raise instead of returning gracefully.
            try:
                tri = Delaunay(pts2d, qhull_options="QJ")  # QJ jitters to avoid precision issues
                triangles = tri.simplices.astype(np.int32)
                triangles = self._prune_long_triangles(points, triangles)
                panel_triangles = self._bridge_triangles(triangles, hand_ranges)
                edges = self._unique_edges(triangles)
                edges = self._prune_long_edges(points, edges)
            except (QhullError, Exception):
                triangles = np.zeros((0, 3), dtype=np.int32)
                edges = []
                panel_triangles = np.zeros((0, 3), dtype=np.int32)

        return {
            "points": points,
            "triangles": triangles,
            "panel_triangles": panel_triangles,
            "edges": edges,
            "skeleton": skeleton,
            "hand_ranges": hand_ranges,
        }


    @staticmethod
    def _bridge_triangles(triangles: np.ndarray, hand_ranges: List[Tuple[str, int, int]]) -> np.ndarray:
        """Keep only triangles that span multiple hands.

        The frosted texture is meant to represent a thrown/formed object or
        surface between the hands. Intra-hand triangles are useful for subtle
        edge structure, but filling them makes the whole hand look textured.
        """
        if len(hand_ranges) < 2 or len(triangles) == 0:
            return np.zeros((0, 3), dtype=np.int32)

        index_to_hand = {}
        for hand_id, (_, start, end) in enumerate(hand_ranges):
            for idx in range(start, end):
                index_to_hand[idx] = hand_id

        kept = []
        for tri in triangles:
            owners = {index_to_hand.get(int(idx)) for idx in tri}
            owners.discard(None)
            if len(owners) >= 2:
                kept.append(tuple(int(idx) for idx in tri))
        return np.asarray(kept, dtype=np.int32) if kept else np.zeros((0, 3), dtype=np.int32)

    def _prune_long_triangles(self, points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        if self.max_edge_length <= 0 or len(triangles) == 0:
            return triangles
        max_len_sq = self.max_edge_length * self.max_edge_length
        kept = []
        for tri in triangles:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            valid = True
            for i, j in ((i0, i1), (i1, i2), (i2, i0)):
                dx = float(points[i, 0] - points[j, 0])
                dy = float(points[i, 1] - points[j, 1])
                if dx * dx + dy * dy > max_len_sq:
                    valid = False
                    break
            if valid:
                kept.append((i0, i1, i2))
        return np.asarray(kept, dtype=np.int32) if kept else np.zeros((0, 3), dtype=np.int32)

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
