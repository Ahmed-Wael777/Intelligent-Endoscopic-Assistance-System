"""
processing.py
-------------
Image processing pipeline for the Smart Endoscopic Assistance System.

Implements the BONUS section requirements:

A) Noise Reduction
   ▸ Bilateral Filter  (chosen: edge-preserving, excellent for tissue detail)
   ▸ Gaussian Filter   (alternative, faster)
   ▸ Median Filter     (salt-and-pepper noise)
   ▸ Non-Local Means   (high quality, slower)

B) Contrast Enhancement
   ▸ CLAHE (preferred for endoscopy: avoids over-amplification of noise
     in dark regions, preserves local tissue contrast)
   ▸ Histogram Equalisation (global, simpler)
   ▸ Adaptive Gamma Correction

C) See feature_extraction.py for feature extraction.

Engineering rationale is included in docstrings for use in oral defence.
"""

import cv2
import numpy as np
from utils import get_logger

logger = get_logger("Processing")

# ──────────────────────────────────────────────────────────────────────────────
# Filter modes (string constants used by GUI)
# ──────────────────────────────────────────────────────────────────────────────
NOISE_METHODS = [
    "Bilateral",          # default
    "Gaussian",
    "Median",
    "Non-Local Means",
    "None",
]

CONTRAST_METHODS = [
    "CLAHE",              # default
    "Histogram EQ",
    "Adaptive Gamma",
    "None",
]


# ──────────────────────────────────────────────────────────────────────────────
# A) NOISE REDUCTION
# ──────────────────────────────────────────────────────────────────────────────

def reduce_noise_bilateral(frame: np.ndarray,
                           d: int = 9,
                           sigma_colour: float = 75,
                           sigma_space:  float = 75) -> np.ndarray:
    """
    Bilateral filter.

    Why chosen:
    -----------
    Endoscopic images contain both Gaussian noise (from the CMOS sensor)
    and speckle. A bilateral filter smooths flat regions while preserving
    sharp mucosal folds and lesion edges — critical for diagnosis.
    Unlike Gaussian blur it does NOT smear tissue boundaries.

    Parameters
    ----------
    d              : diameter of pixel neighbourhood
    sigma_colour   : filter sigma in colour space (range of colours considered)
    sigma_space    : filter sigma in coordinate space (spatial extent)
    """
    return cv2.bilateralFilter(frame, d, sigma_colour, sigma_space)


def reduce_noise_gaussian(frame: np.ndarray,
                          ksize: int = 5,
                          sigma: float = 1.2) -> np.ndarray:
    """
    Gaussian blur.  Fast but blurs edges.  Good for heavily noisy frames.
    """
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(frame, (k, k), sigma)


def reduce_noise_median(frame: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Median filter.  Ideal for impulsive (salt-and-pepper) noise from
    fibre-bundle CCD sensors or video compression artefacts.
    """
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(frame, k)


def reduce_noise_nlm(frame: np.ndarray,
                     h: float = 10,
                     template_window: int = 7,
                     search_window:   int = 21) -> np.ndarray:
    """
    Non-Local Means denoising.  Best quality; computationally expensive.
    Averages similar patches across the whole image — excellent for
    preserving fine mucosal texture while suppressing noise.
    """
    return cv2.fastNlMeansDenoisingColored(frame,
                                           None, h, h,
                                           template_window,
                                           search_window)


def reduce_noise(frame: np.ndarray, method: str = "Bilateral") -> np.ndarray:
    """
    Dispatcher: apply selected noise-reduction method.
    """
    m = method.strip()
    if m == "Bilateral":      return reduce_noise_bilateral(frame)
    if m == "Gaussian":       return reduce_noise_gaussian(frame)
    if m == "Median":         return reduce_noise_median(frame)
    if m == "Non-Local Means":return reduce_noise_nlm(frame)
    return frame.copy()    # "None"


# ──────────────────────────────────────────────────────────────────────────────
# B) CONTRAST ENHANCEMENT
# ──────────────────────────────────────────────────────────────────────────────

def enhance_contrast_clahe(frame: np.ndarray,
                           clip_limit:   float = 2.0,
                           tile_grid:    tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Why chosen:
    -----------
    Standard global histogram equalisation often produces over-saturated
    or artefact-prone results in endoscopic images where illumination is
    inherently uneven (bright specular spot, dark peripheral vignette).
    CLAHE divides the image into tiles and equalises each locally with a
    clip limit that prevents over-amplification of noise.  This is the
    method of choice in clinical endoscopy AI pipelines.

    Applied on the L channel of LAB colour space so hue is preserved.

    Parameters
    ----------
    clip_limit  : threshold for contrast amplification (2–4 typical)
    tile_grid   : number of tiles (rows × cols)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq  = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_contrast_histeq(frame: np.ndarray) -> np.ndarray:
    """
    Global Histogram Equalisation on the Value channel (HSV).
    Simpler than CLAHE; can over-saturate very bright or dark regions.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)


def enhance_contrast_gamma(frame: np.ndarray) -> np.ndarray:
    """
    Adaptive Gamma Correction.
    Computes mean luminance and adjusts gamma to target 128 (mid-grey).
    Boosts dark images without washing out bright ones.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = float(gray.mean())
    if mean_lum < 1:
        return frame.copy()
    # gamma = log(128) / log(mean)
    gamma = np.log(128.0) / (np.log(mean_lum) + 1e-6)
    gamma = float(np.clip(gamma, 0.4, 3.0))
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma)
    lut = np.clip(lut * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(frame, lut)


def enhance_contrast(frame: np.ndarray, method: str = "CLAHE") -> np.ndarray:
    """
    Dispatcher: apply selected contrast-enhancement method.
    """
    m = method.strip()
    if m == "CLAHE":          return enhance_contrast_clahe(frame)
    if m == "Histogram EQ":   return enhance_contrast_histeq(frame)
    if m == "Adaptive Gamma": return enhance_contrast_gamma(frame)
    return frame.copy()    # "None"


# ──────────────────────────────────────────────────────────────────────────────
# Full processing pipeline (noise → contrast)
# ──────────────────────────────────────────────────────────────────────────────
class ImageProcessor:
    """
    Stateful wrapper that applies noise reduction then contrast enhancement.
    Settings can be changed at runtime via the GUI.
    """

    def __init__(self,
                 noise_method:    str = "Bilateral",
                 contrast_method: str = "CLAHE"):
        self.noise_method    = noise_method
        self.contrast_method = contrast_method

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            from utils import blank_frame
            return blank_frame()
        denoised  = reduce_noise(frame,    self.noise_method)
        enhanced  = enhance_contrast(denoised, self.contrast_method)
        return enhanced

    def status_dict(self) -> dict:
        return {
            "noise_method":    self.noise_method,
            "contrast_method": self.contrast_method,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Side-by-side comparison helper (used in captured-image report)
# ──────────────────────────────────────────────────────────────────────────────
def make_comparison(original: np.ndarray,
                    processed: np.ndarray,
                    label_orig: str = "Original",
                    label_proc: str = "Processed") -> np.ndarray:
    """
    Create a horizontal before/after comparison image with labels.
    Both images are resized to the same height if necessary.
    """
    h = max(original.shape[0], processed.shape[0])
    w = original.shape[1]

    def pad(img):
        ih, iw = img.shape[:2]
        if ih < h:
            pad_bottom = h - ih
            img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(20, 20, 40))
        return cv2.resize(img, (w, h))

    orig_r = pad(original.copy())
    proc_r = pad(processed.copy())

    # Draw labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, lbl in ((orig_r, label_orig), (proc_r, label_proc)):
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, lbl, (6, 17), font, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

    # Divider
    divider = np.full((h, 3, 3), 80, dtype=np.uint8)
    return np.hstack([orig_r, divider, proc_r])
