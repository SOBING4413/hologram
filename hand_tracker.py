"""
hand_tracker.py
---------------
Thin wrapper around MediaPipe Hands. Detects up to two hands per frame and
returns, for each detected hand:
    - "label"      : "Left" or "Right" (as classified by MediaPipe)
    - "score"      : handedness confidence
    - "landmarks"  : list of 21 (x, y, z) tuples, normalized to [0, 1]
                      (x, y relative to image width/height, z relative depth)

Keeping this in its own module means main.py / renderer.py never touch the
MediaPipe API directly.
"""

import sys
from typing import Dict, List

import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.6,
                 tracking_confidence: float = 0.6):
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            # This should already have been caught by env_check.run_all() in
            # main.py, but HandTracker can also be imported/used standalone,
            # so fail with an actionable message instead of a bare
            # AttributeError.
            raise RuntimeError(
                "mediapipe.solutions.hands tidak tersedia di instalasi "
                f"mediapipe ini (Python {sys.version.split()[0]}, mediapipe "
                f"{getattr(mp, '__version__', 'unknown')}).\n"
                "Jalankan 'python env_check.py' untuk diagnosa lengkap dan "
                "perbaikan otomatis, atau lihat README.md bagian "
                "Troubleshooting."
            )

        self._mp_hands = mp.solutions.hands
        self.connections = list(self._mp_hands.HAND_CONNECTIONS)

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame_rgb) -> List[Dict]:
        """
        frame_rgb: HxWx3 uint8 numpy array, RGB order (already mirrored if
                   you want mirrored tracking to match the mirrored display).
        Returns a list (len 0..max_hands) of dicts as described above.
        """
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        hands: List[Dict] = []
        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if i < len(handedness_list):
                    classification = handedness_list[i].classification[0]
                    label = classification.label
                    score = classification.score
                else:
                    label = f"Hand{i}"
                    score = 0.0

                pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                hands.append({"label": label, "score": score, "landmarks": pts})
        return hands

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
