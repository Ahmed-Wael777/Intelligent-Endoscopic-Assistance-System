"""
utils.py
--------
Shared constants, helper functions, and lightweight logging utilities
for the Smart Endoscopic Assistance System.
"""

import os
import logging
import datetime
import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Project-wide constants
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR      = os.path.join(PROJECT_ROOT, "Captured_Images")
SAMPLE_DATA_DIR   = os.path.join(PROJECT_ROOT, "sample_data")

DISPLAY_W, DISPLAY_H = 640, 480   # canonical display resolution
PANEL_W,   PANEL_H   = 320, 240   # small panel thumbnails

# Colour palette (BGR for OpenCV, hex for Tkinter)
COLOUR = {
    "bg_dark":    "#1a1a2e",
    "bg_mid":     "#16213e",
    "bg_panel":   "#0f3460",
    "accent":     "#e94560",
    "accent2":    "#00d2ff",
    "text_light": "#e0e0e0",
    "text_dim":   "#8892b0",
    "green":      "#64ffda",
    "yellow":     "#ffd700",
    "red":        "#ff6b6b",
    "white":      "#ffffff",
}

# ──────────────────────────────────────────────────────────────────────────────
# Directory setup
# ──────────────────────────────────────────────────────────────────────────────
def ensure_directories() -> None:
    """Create required output directories if they do not exist."""
    for d in (CAPTURED_DIR, SAMPLE_DATA_DIR):
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def get_logger(name: str = "SmartEndoscope") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()


# ──────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ──────────────────────────────────────────────────────────────────────────────
def timestamp_str() -> str:
    """Return a filesystem-safe timestamp string: YYYYMMDD_HHMMSS"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_display() -> str:
    """Return a human-readable timestamp for on-screen overlay."""
    return datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────
def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to (width, height) using INTER_LINEAR."""
    if frame is None:
        return blank_frame(width, height)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def blank_frame(width: int = DISPLAY_W, height: int = DISPLAY_H,
                colour: tuple = (20, 20, 40)) -> np.ndarray:
    """Return a solid-colour blank BGR frame."""
    f = np.zeros((height, width, 3), dtype=np.uint8)
    f[:] = colour
    return f


def overlay_text(frame: np.ndarray, text: str,
                 pos: tuple = (10, 30),
                 font_scale: float = 0.55,
                 colour: tuple = (0, 255, 200),
                 thickness: int = 1,
                 shadow: bool = True) -> np.ndarray:
    """
    Draw text with an optional dark shadow for readability on any background.
    Works in-place and also returns the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1), font,
                    font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font,
                font_scale, colour, thickness, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, brightness: float, nav_label: str,
             paused: bool, fps: float) -> np.ndarray:
    """
    Draw the Heads-Up Display overlay on a frame:
    timestamp, brightness bar, navigation status, pause indicator, FPS.
    Returns a *copy* so the original is unmodified.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # ── semi-transparent top bar ──────────────────────────────────────────────
    bar = out[0:36, :].copy()
    cv2.rectangle(bar, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, out[0:36, :], 0.45, 0, out[0:36, :])

    # Timestamp
    overlay_text(out, timestamp_display(), (8, 22), 0.48, (200, 230, 255))

    # FPS (top-right)
    fps_txt = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    overlay_text(out, fps_txt, (w - tw - 8, 22), 0.48, (100, 255, 180))

    # ── brightness bar (bottom-left) ─────────────────────────────────────────
    bx, by, bw, bh = 8, h - 28, 140, 12
    cv2.rectangle(out, (bx - 1, by - 1), (bx + bw + 1, by + bh + 1),
                  (80, 80, 80), 1)
    filled = int(bw * brightness / 100.0)
    cv2.rectangle(out, (bx, by), (bx + filled, by + bh), (0, 200, 255), -1)
    overlay_text(out, f"Brightness: {int(brightness)}%",
                 (bx, h - 34), 0.42, (220, 220, 220))

    # ── navigation label (bottom-right) ──────────────────────────────────────
    nav_colour = (50, 255, 150) if nav_label != "Stationary" else (160, 160, 160)
    (nw, _), _ = cv2.getTextSize(nav_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    overlay_text(out, nav_label, (w - nw - 8, h - 10), 0.52, nav_colour)

    # ── pause badge ──────────────────────────────────────────────────────────
    if paused:
        cv2.rectangle(out, (w // 2 - 55, h // 2 - 22),
                      (w // 2 + 55, h // 2 + 22), (0, 0, 0), -1)
        overlay_text(out, "  PAUSED", (w // 2 - 48, h // 2 + 8),
                     0.75, (0, 200, 255), 2)

    return out


def save_image(frame: np.ndarray, prefix: str = "capture") -> str:
    """
    Save a BGR frame to Captured_Images/ with a timestamped filename.
    Returns the full path.
    """
    ensure_directories()
    fname = f"{prefix}_{timestamp_str()}.png"
    path  = os.path.join(CAPTURED_DIR, fname)
    cv2.imwrite(path, frame)
    logger.info("Image saved → %s", path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# BGR ↔ RGB conversion (for PIL / Tkinter)
# ──────────────────────────────────────────────────────────────────────────────
def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
