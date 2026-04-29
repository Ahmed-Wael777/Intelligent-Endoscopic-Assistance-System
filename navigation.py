"""
navigation.py
-------------
Simulates the Insertion Tube & Navigation subsystem of the endoscope.

Supported movement directions
──────────────────────────────
  UP / DOWN / LEFT / RIGHT   → translate the viewport (frame shift)
  FORWARD                    → zoom in  (simulate advancing the tube)
  BACKWARD                   → zoom out (simulate retracting the tube)

The NavigationController applies affine/zoom transformations to produce
the "camera-in-motion" illusion without requiring real hardware.
A small directional arrow HUD is also drawn on the frame.
"""

import time
import cv2
import numpy as np
from utils import get_logger, DISPLAY_W, DISPLAY_H

logger = get_logger("Navigation")

# ──────────────────────────────────────────────────────────────────────────────
# Direction constants
# ──────────────────────────────────────────────────────────────────────────────
DIRECTIONS = {
    "UP":       ( 0, -1,  0),   # (dx, dy, dz)
    "DOWN":     ( 0,  1,  0),
    "LEFT":     (-1,  0,  0),
    "RIGHT":    ( 1,  0,  0),
    "FORWARD":  ( 0,  0,  1),
    "BACKWARD": ( 0,  0, -1),
    "NONE":     ( 0,  0,  0),
}

DIRECTION_LABELS = {
    "UP":       "Moving Up",
    "DOWN":     "Moving Down",
    "LEFT":     "Turning Left",
    "RIGHT":    "Turning Right",
    "FORWARD":  "Moving Forward",
    "BACKWARD": "Retracting",
    "NONE":     "Stationary",
}

# Arrow shapes for on-frame HUD (relative pixel offsets from centre)
_ARROW_SHAPES = {
    "UP":       [(0, -18), (-10, 0), (10, 0)],
    "DOWN":     [(0,  18), (-10, 0), (10, 0)],
    "LEFT":     [(-18, 0), (0, -10), (0, 10)],
    "RIGHT":    [( 18, 0), (0, -10), (0, 10)],
    "FORWARD":  None,   # rendered as concentric circles (zoom-in symbol)
    "BACKWARD": None,
    "NONE":     None,
}

# ──────────────────────────────────────────────────────────────────────────────
class NavigationController:
    """
    Maintains the virtual camera pose and applies it to frames.

    State
    -----
    offset_x, offset_y  :  pixel shift of the viewport
    zoom_level          :  multiplicative zoom (1.0 = no zoom)
    current_direction   :  last commanded direction key
    """

    # ── tuneable parameters ───────────────────────────────────────────────────
    STEP_XY        = 8          # pixels per keypress / button click
    STEP_Z         = 0.04       # zoom delta per step
    ZOOM_MIN       = 0.5
    ZOOM_MAX       = 3.0
    MAX_OFFSET_X   = 200
    MAX_OFFSET_Y   = 150
    DECAY_RATE     = 0.82       # inertia: offset damps to zero when key released
    DIRECTION_HOLD = 0.45       # seconds direction label stays visible

    def __init__(self):
        self.offset_x          : int   = 0
        self.offset_y          : int   = 0
        self.zoom_level        : float = 1.0
        self.current_direction : str   = "NONE"
        self._direction_ts     : float = 0.0   # last time direction changed
        self._continuous_move  : bool  = False  # True while key is held
        logger.info("NavigationController initialised.")

    # ── movement commands ─────────────────────────────────────────────────────
    def move(self, direction: str, continuous: bool = False):
        """
        Apply one step in *direction*.
        Call repeatedly while a button/key is held (continuous=True).
        """
        direction = direction.upper()
        if direction not in DIRECTIONS:
            return
        dx, dy, dz = DIRECTIONS[direction]
        self.offset_x = int(np.clip(self.offset_x + dx * self.STEP_XY,
                                    -self.MAX_OFFSET_X, self.MAX_OFFSET_X))
        self.offset_y = int(np.clip(self.offset_y + dy * self.STEP_XY,
                                    -self.MAX_OFFSET_Y, self.MAX_OFFSET_Y))
        self.zoom_level = float(np.clip(self.zoom_level + dz * self.STEP_Z,
                                        self.ZOOM_MIN, self.ZOOM_MAX))
        self.current_direction = direction
        self._direction_ts     = time.time()
        self._continuous_move  = continuous

    def stop_movement(self):
        """Call when a navigation key / button is released."""
        self.current_direction = "NONE"
        self._continuous_move  = False

    def reset(self):
        """Reset pose to origin."""
        self.offset_x          = 0
        self.offset_y          = 0
        self.zoom_level        = 1.0
        self.current_direction = "NONE"
        logger.info("Navigation reset to origin.")

    # ── inertial damping (call each frame if desired) ─────────────────────────
    def apply_inertia(self):
        """
        Gradually return offset to zero when no key is held.
        Creates a smooth deceleration feel.
        """
        if not self._continuous_move:
            self.offset_x = int(self.offset_x * self.DECAY_RATE)
            self.offset_y = int(self.offset_y * self.DECAY_RATE)

    # ── frame transformation ──────────────────────────────────────────────────
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Warp *frame* according to current pose (offset + zoom).
        Returns a new frame of the same size.
        """
        if frame is None or frame.size == 0:
            from utils import blank_frame
            return blank_frame()

        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Build 2×3 affine matrix: zoom about centre + translate
        zoom = self.zoom_level
        M = np.array([
            [zoom, 0,    cx * (1 - zoom) + self.offset_x],
            [0,    zoom, cy * (1 - zoom) + self.offset_y],
        ], dtype=np.float64)

        warped = cv2.warpAffine(frame, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        return warped

    # ── HUD overlay ───────────────────────────────────────────────────────────
    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw navigation compass rose and current direction label.
        Returns a copy of the frame with the overlay.
        """
        out   = frame.copy()
        h, w  = out.shape[:2]

        # Compass rose background (top-right)
        cx, cy = w - 55, 70
        cv2.circle(out, (cx, cy), 44, (20, 20, 20), -1)
        cv2.circle(out, (cx, cy), 44, (80, 80, 80),  1)

        direction_active = (time.time() - self._direction_ts) < self.DIRECTION_HOLD
        active_dir = self.current_direction if direction_active else "NONE"

        # Draw four directional triangles
        compass = {
            "UP":    [(cx,     cy-32), (cx-9,  cy-16), (cx+9,  cy-16)],
            "DOWN":  [(cx,     cy+32), (cx-9,  cy+16), (cx+9,  cy+16)],
            "LEFT":  [(cx-32,  cy),    (cx-16, cy-9),   (cx-16, cy+9)],
            "RIGHT": [(cx+32,  cy),    (cx+16, cy-9),   (cx+16, cy+9)],
        }
        for d, pts in compass.items():
            colour = (0, 220, 255) if d == active_dir else (90, 90, 90)
            cv2.fillPoly(out, [np.array(pts, np.int32)], colour)

        # FORWARD / BACKWARD circles
        fwd_colour = (0, 220, 255) if active_dir == "FORWARD"  else (80, 80, 80)
        bwd_colour = (0, 220, 255) if active_dir == "BACKWARD" else (80, 80, 80)
        cv2.circle(out, (cx, cy), 10,  fwd_colour, 2)
        cv2.circle(out, (cx, cy), 5,   fwd_colour, -1)
        cv2.circle(out, (cx, cy), 18, bwd_colour,  1)

        # Zoom level indicator
        zoom_txt = f"x{self.zoom_level:.2f}"
        cv2.putText(out, zoom_txt, (cx - 18, cy + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        # Direction label (bottom of compass)
        lbl = DIRECTION_LABELS.get(active_dir, "")
        cv2.putText(out, lbl, (8, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (50, 255, 150) if active_dir != "NONE" else (140, 140, 140),
                    1, cv2.LINE_AA)

        return out

    # ── status dict ───────────────────────────────────────────────────────────
    def status_dict(self) -> dict:
        return {
            "offset_x":  self.offset_x,
            "offset_y":  self.offset_y,
            "zoom":      round(self.zoom_level, 3),
            "direction": self.current_direction,
        }

    @property
    def label(self) -> str:
        return DIRECTION_LABELS.get(self.current_direction, "Stationary")
