"""
main.py
-------
Entry point for the Smart Endoscopic Assistance System.
Handles environment validation and launches the GUI.

Run:
    python main.py
"""

import sys
import os

# ── Ensure we can import project modules ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Validate Python version ───────────────────────────────────────────────────
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or newer is required.")

# ── Check for sample data; generate if missing ───────────────────────────────
from utils import SAMPLE_DATA_DIR, ensure_directories, get_logger
import glob

logger = get_logger("Main")
ensure_directories()

sample_frames = glob.glob(os.path.join(SAMPLE_DATA_DIR, "endo_*.jpg"))
if not sample_frames:
    logger.info("Sample frames not found – generating synthetic data...")
    try:
        import generate_samples   # runs __main__ block implicitly? No – call directly
        import importlib, subprocess
        subprocess.run([sys.executable,
                        os.path.join(PROJECT_ROOT, "generate_samples.py")],
                       check=True)
    except Exception as exc:
        logger.error("Could not generate sample data: %s", exc)
        sys.exit("Sample data generation failed. "
                 "Run 'python generate_samples.py' manually.")

# ── Validate critical third-party imports ─────────────────────────────────────
REQUIRED = {
    "cv2":           "opencv-python",
    "numpy":         "numpy",
    "PIL":           "Pillow",
    "skimage":       "scikit-image",
    "matplotlib":    "matplotlib",
}
missing = []
for module, pkg in REQUIRED.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(pkg)

if missing:
    sys.exit(
        f"Missing dependencies: {', '.join(missing)}\n"
        f"Install with:  pip install {' '.join(missing)}"
    )

# ── Launch GUI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # On some Linux/X11 systems Tkinter needs the DISPLAY variable
    if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
        os.environ.setdefault("DISPLAY", ":0")

    from gui import SmartEndoscopeApp
    logger.info("Launching Smart Endoscopic Assistance System...")
    app = SmartEndoscopeApp()
    app.mainloop()
