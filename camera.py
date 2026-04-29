"""
camera.py
---------
Manages the imaging source for the Smart Endoscopic Assistance System.

Priority order:
  1. Live webcam (index 0)
  2. Pre-recorded video file (sample_data/*.mp4 or .avi)
  3. Synthetic PNG image sequence (sample_data/endo_*.png)   ← default demo

All sources expose the same interface:
    camera.read()  →  (success: bool, frame: np.ndarray)
    camera.stop()
"""

import os
import glob
import threading
import time

import cv2
import numpy as np

from utils import (SAMPLE_DATA_DIR, DISPLAY_W, DISPLAY_H,
                   blank_frame, resize_frame, get_logger)

logger = get_logger("Camera")


# ──────────────────────────────────────────────────────────────────────────────
# Image-sequence source (default; no hardware required)
# ──────────────────────────────────────────────────────────────────────────────
class ImageSequenceCamera:
    """
    Loops through PNG frames in sample_data/ at ~25 FPS (simulated).
    Thread-safe; exposes the same read() / stop() interface as OpenCV VideoCapture.
    """

    TARGET_FPS = 25

    def __init__(self, directory: str = SAMPLE_DATA_DIR):
        pattern = os.path.join(directory, "endo_*.jpg")
        self._paths = sorted(glob.glob(pattern))
        if not self._paths:
            raise FileNotFoundError(
                f"No endo_*.jpg frames found in '{directory}'. "
                "Run generate_samples.py first."
            )
        self._idx     = 0
        self._lock    = threading.Lock()
        self._running = True
        self._current = cv2.imread(self._paths[0])
        self._thread  = threading.Thread(target=self._advance_loop,
                                         daemon=True, name="CamAdvance")
        self._thread.start()
        logger.info("ImageSequenceCamera → %d frames from %s",
                    len(self._paths), directory)

    # ── background thread: advance frame at TARGET_FPS ─────────────────────
    def _advance_loop(self):
        interval = 1.0 / self.TARGET_FPS
        while self._running:
            t0 = time.perf_counter()
            next_idx = (self._idx + 1) % len(self._paths)
            frame = cv2.imread(self._paths[next_idx])
            if frame is not None:
                with self._lock:
                    self._idx     = next_idx
                    self._current = frame
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, interval - elapsed))

    # ── public interface ─────────────────────────────────────────────────────
    def read(self):
        with self._lock:
            frame = self._current.copy() if self._current is not None else None
        if frame is None:
            return False, blank_frame()
        return True, resize_frame(frame, DISPLAY_W, DISPLAY_H)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        logger.info("ImageSequenceCamera stopped.")

    @property
    def fps(self) -> float:
        return float(self.TARGET_FPS)

    @property
    def source_label(self) -> str:
        return f"Synthetic sequence ({len(self._paths)} frames)"


# ──────────────────────────────────────────────────────────────────────────────
# Webcam / video-file source wrapper
# ──────────────────────────────────────────────────────────────────────────────
class VideoCaptureCamera:
    """
    Wraps cv2.VideoCapture for a webcam index or video file path.
    Loops video files automatically.
    """

    def __init__(self, source):
        self._source = source
        self._cap    = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self._is_file = isinstance(source, str)
        logger.info("VideoCaptureCamera → source=%s", source)

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            if self._is_file:          # loop
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok or frame is None:
                return False, blank_frame()
        return True, resize_frame(frame, DISPLAY_W, DISPLAY_H)

    def stop(self):
        self._cap.release()
        logger.info("VideoCaptureCamera released.")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def source_label(self) -> str:
        return f"Video: {self._source}"


# ──────────────────────────────────────────────────────────────────────────────
# Factory: auto-select best available source
# ──────────────────────────────────────────────────────────────────────────────
def create_camera(prefer_webcam: bool = True):
    """
    Auto-selects the imaging source:
      1. Webcam (if prefer_webcam=True and a device is found)
      2. Video file in sample_data/
      3. PNG image sequence in sample_data/
    Returns an object with .read(), .stop(), .fps, .source_label.
    """
    if prefer_webcam:
        try:
            cap = cv2.VideoCapture(0)
            ok, _ = cap.read()
            cap.release()
            if ok:
                logger.info("Using live webcam (index 0)")
                return VideoCaptureCamera(0)
        except Exception as exc:
            logger.warning("Webcam probe failed: %s", exc)

    # Try video file
    vid_patterns = ["*.mp4", "*.avi", "*.mov"]
    for pat in vid_patterns:
        hits = sorted(glob.glob(os.path.join(SAMPLE_DATA_DIR, pat)))
        if hits:
            logger.info("Using video file: %s", hits[0])
            return VideoCaptureCamera(hits[0])

    # Fallback: PNG sequence
    logger.info("Using synthetic PNG image sequence.")
    return ImageSequenceCamera(SAMPLE_DATA_DIR)
