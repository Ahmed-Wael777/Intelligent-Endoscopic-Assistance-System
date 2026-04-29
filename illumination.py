"""
illumination.py
---------------
Simulates the LED illumination system of an endoscope.

The IlluminationController:
  - Maintains a brightness level (0–100 %).
  - Applies a physically-motivated brightness + colour-temperature model
    to every frame using cv2.convertScaleAbs().
  - At low brightness the image also warms slightly (reddish tint)
    to mimic reduced white LED output, and a vignette deepens.
  - At high brightness mild specular bloom is added.
"""

import cv2
import numpy as np
from utils import get_logger

logger = get_logger("Illumination")

# ──────────────────────────────────────────────────────────────────────────────
# Pre-computed vignette mask (cached at module load)
# ──────────────────────────────────────────────────────────────────────────────
_W, _H = 640, 480
_cx, _cy = _W // 2, _H // 2
_Y, _X = np.ogrid[:_H, :_W]
_dist = np.sqrt((_X - _cx) ** 2 + (_Y - _cy) ** 2).astype(np.float32)
_MAX_DIST = float(np.sqrt(_cx ** 2 + _cy ** 2))
_BASE_VIGNETTE: np.ndarray = np.clip(1.0 - 0.75 * (_dist / _MAX_DIST) ** 2,
                                     0.0, 1.0)


class IlluminationController:
    """
    Encapsulates LED illumination simulation.

    Usage
    -----
    ctrl = IlluminationController(initial_brightness=70)
    bright_frame = ctrl.apply(raw_frame)
    ctrl.set_brightness(85)
    """

    # Mapping from brightness% to cv2.convertScaleAbs alpha (gain) & beta (bias)
    _MIN_ALPHA, _MAX_ALPHA = 0.08, 1.85   # very dark → slightly over-exposed
    _MIN_BETA,  _MAX_BETA  = -60,  30     # shadow lift at low brightness

    def __init__(self, initial_brightness: float = 70.0):
        self._brightness: float = float(np.clip(initial_brightness, 0, 100))
        logger.info("IlluminationController initialised  brightness=%.1f%%",
                    self._brightness)

    # ── properties ─────────────────────────────────────────────────────────────
    @property
    def brightness(self) -> float:
        return self._brightness

    @brightness.setter
    def brightness(self, value: float):
        self._brightness = float(np.clip(value, 0, 100))

    def set_brightness(self, value: float):
        self.brightness = value

    # ── private helpers ────────────────────────────────────────────────────────
    def _alpha_beta(self) -> tuple:
        """Compute (alpha, beta) for cv2.convertScaleAbs from brightness%."""
        t = self._brightness / 100.0          # normalised [0,1]
        alpha = self._MIN_ALPHA + t * (self._MAX_ALPHA - self._MIN_ALPHA)
        beta  = self._MIN_BETA  + t * (self._MAX_BETA  - self._MIN_BETA)
        return alpha, beta

    def _colour_temperature_shift(self, frame: np.ndarray) -> np.ndarray:
        """
        At low brightness LEDs appear warmer (more red/green, less blue).
        At high brightness slight cool blue boost simulates xenon/white LED.
        Applied as a per-channel multiplicative bias.
        """
        t = self._brightness / 100.0
        # Channel multipliers  (B, G, R)
        blue_m  = 0.65 + 0.50 * t          # 0.65 → 1.15
        green_m = 0.85 + 0.25 * t          # 0.85 → 1.10
        red_m   = 1.10 - 0.15 * t          # 1.10 → 0.95
        out = frame.astype(np.float32)
        out[:, :, 0] = np.clip(out[:, :, 0] * blue_m,  0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * green_m, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * red_m,   0, 255)
        return out.astype(np.uint8)

    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """
        Deepen the vignette at lower brightness (less light reaches edges).
        """
        t = self._brightness / 100.0
        strength = 0.55 + (1.0 - t) * 0.40     # 0.55 (bright) → 0.95 (dark)
        h, w = frame.shape[:2]
        if (h, w) != (_H, _W):
            import cv2 as _cv2
            mask = _cv2.resize(_BASE_VIGNETTE, (w, h),
                               interpolation=_cv2.INTER_LINEAR)
        else:
            mask = _BASE_VIGNETTE

        dynamic_mask = np.clip(1.0 - strength * (_dist[:h, :w] / _MAX_DIST) ** 2,
                               0.0, 1.0)
        out = frame.astype(np.float32)
        for c in range(3):
            out[:, :, c] *= dynamic_mask
        return np.clip(out, 0, 255).astype(np.uint8)

    def _bloom(self, frame: np.ndarray) -> np.ndarray:
        """
        Subtle specular bloom at brightness > 80 %: mimics light scatter.
        """
        if self._brightness < 80:
            return frame
        t  = (self._brightness - 80) / 20.0        # 0 → 1 above 80 %
        blurred = cv2.GaussianBlur(frame, (21, 21), 8)
        return cv2.addWeighted(frame, 1.0, blurred, 0.18 * t, 0).astype(np.uint8)

    # ── public API ─────────────────────────────────────────────────────────────
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the complete illumination pipeline to *frame* (BGR uint8).
        Returns a new frame; original is not modified.

        Pipeline:
          1. cv2.convertScaleAbs  (gain + bias)
          2. colour-temperature shift
          3. vignette
          4. specular bloom (high brightness only)
        """
        if frame is None or frame.size == 0:
            from utils import blank_frame
            return blank_frame()

        alpha, beta = self._alpha_beta()

        # Step 1 – brightness gain + bias
        lit = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # Step 2 – colour temperature
        lit = self._colour_temperature_shift(lit)

        # Step 3 – vignette
        lit = self._apply_vignette(lit)

        # Step 4 – bloom
        lit = self._bloom(lit)

        return lit

    # ── convenience ────────────────────────────────────────────────────────────
    def status_dict(self) -> dict:
        return {
            "brightness_pct": round(self._brightness, 1),
            "alpha": round(self._alpha_beta()[0], 3),
            "beta":  round(self._alpha_beta()[1], 1),
        }
