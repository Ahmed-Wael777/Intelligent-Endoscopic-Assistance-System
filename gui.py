"""
gui.py
------
Full Tkinter GUI for the Smart Endoscopic Assistance System.
Implements the Display & Output System and Control System subsystems.

Layout
──────
┌─────────────────────────────────────────────────────────────┐
│  TITLE BAR                            [status bar right]    │
├────────────────────┬────────────────────┬───────────────────┤
│  LIVE VIEW         │  PROCESSED VIEW    │  FEATURE PANEL    │
│  (main camera)     │  (enhanced)        │  (text features)  │
├────────────────────┴────────────────────┼───────────────────┤
│  CONTROL PANEL (left)                   │  CAPTURE PREVIEW  │
│   • Illumination slider                 │                   │
│   • Camera start/stop                  │                   │
│   • Capture / Reset / Pause / Exit     │                   │
│   • Navigation arrows                  │                   │
│   • Processing method selectors        │                   │
└─────────────────────────────────────────┴───────────────────┘
"""

import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from utils      import (COLOUR, DISPLAY_W, DISPLAY_H, PANEL_W, PANEL_H,
                         bgr_to_rgb, save_image, draw_hud, get_logger,
                         timestamp_display, blank_frame)
from camera     import create_camera
from illumination import IlluminationController
from navigation   import NavigationController
from processing   import ImageProcessor, NOISE_METHODS, CONTRAST_METHODS, make_comparison
from feature_extraction import (extract_all_features, render_feature_panel,
                                 draw_contours)

logger = get_logger("GUI")


# ──────────────────────────────────────────────────────────────────────────────
# Splash screen
# ──────────────────────────────────────────────────────────────────────────────
def run_splash() -> None:
    """
    Show a standalone splash window using its own Tk root.
    Fully completes (and destroys the root) before the main app window
    is ever created — avoids the wait_window / bad-window-path crash
    that occurs on Windows when a Toplevel is destroyed inside __init__.
    """
    root = tk.Tk()
    root.overrideredirect(True)
    root.configure(bg=COLOUR["bg_dark"])
    w, h = 520, 260
    root.update_idletasks()          # ensures winfo_screenwidth is valid
    sx = root.winfo_screenwidth()  // 2 - w // 2
    sy = root.winfo_screenheight() // 2 - h // 2
    root.geometry(f"{w}x{h}+{sx}+{sy}")
    root.lift()
    root.attributes("-topmost", True)

    tk.Label(root, text="🔬", font=("Segoe UI Emoji", 36),
             bg=COLOUR["bg_dark"], fg=COLOUR["accent"]).pack(pady=(24, 4))
    tk.Label(root, text="Smart Endoscopic Assistance System",
             font=("Segoe UI", 16, "bold"),
             bg=COLOUR["bg_dark"], fg=COLOUR["text_light"]).pack()
    tk.Label(root, text="Medical Equipment II  ·  SBE3220",
             font=("Segoe UI", 10),
             bg=COLOUR["bg_dark"], fg=COLOUR["text_dim"]).pack(pady=2)
    status_lbl = tk.Label(root, text="Initialising...",
                          font=("Segoe UI", 9, "italic"),
                          bg=COLOUR["bg_dark"], fg=COLOUR["accent2"])
    status_lbl.pack(pady=6)
    pbar = ttk.Progressbar(root, length=380, mode="determinate", maximum=100)
    pbar.pack(pady=8)
    root.update()

    msgs = [
        "Loading camera subsystem...",
        "Initialising illumination controller...",
        "Loading navigation engine...",
        "Setting up image processing pipeline...",
        "Building feature extractor...",
        "Launching GUI...",
    ]
    step = 100 // len(msgs)
    for i, msg in enumerate(msgs):
        status_lbl.config(text=msg)
        pbar["value"] = min(100, (i + 1) * step)
        root.update()
        time.sleep(0.28)

    root.destroy()   # cleanly destroys the splash Tk root before main app starts


# ──────────────────────────────────────────────────────────────────────────────
# Helper: numpy BGR → Tkinter PhotoImage
# ──────────────────────────────────────────────────────────────────────────────
def _to_photoimage(frame: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    if frame is None or frame.size == 0:
        frame = blank_frame(w, h)
    rgb   = bgr_to_rgb(cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR))
    image = Image.fromarray(rgb)
    return ImageTk.PhotoImage(image)


# ──────────────────────────────────────────────────────────────────────────────
# Main Application Window
# ──────────────────────────────────────────────────────────────────────────────
class SmartEndoscopeApp(tk.Tk):

    UPDATE_MS       = 33    # ~30 fps GUI refresh
    FEATURE_PERIOD  = 8     # extract features every N frames

    def __init__(self):
        super().__init__()

        # ── splash (runs its own Tk root, finishes before this window opens) ──
        run_splash()

        # ── window config ─────────────────────────────────────────────────────
        self.title("Smart Endoscopic Assistance System  |  SBE3220")
        self.configure(bg=COLOUR["bg_dark"])
        self.resizable(True, True)
        self._centre_window(1340, 820)

        # ── subsystems ────────────────────────────────────────────────────────
        self.camera     = create_camera(prefer_webcam=False)
        self.illum      = IlluminationController(initial_brightness=70)
        self.nav        = NavigationController()
        self.processor  = ImageProcessor()

        # ── state flags ───────────────────────────────────────────────────────
        self._running          = True
        self._camera_on        = True
        self._paused           = False
        self._show_contours    = tk.BooleanVar(value=False)
        self._frame_count      = 0
        self._fps_ts           = time.perf_counter()
        self._fps              = 0.0
        self._last_raw         = blank_frame()
        self._last_processed   = blank_frame()
        self._capture_preview  = blank_frame(PANEL_W, PANEL_H)
        self._features: dict   = {}
        self._keys_held: set   = set()

        # ── build UI ──────────────────────────────────────────────────────────
        self._build_menu()
        self._build_layout()
        self._bind_keys()

        # ── start render loop ─────────────────────────────────────────────────
        self.after(self.UPDATE_MS, self._update_loop)
        self.protocol("WM_DELETE_WINDOW", self._on_exit)
        logger.info("Application started.")

    # ══════════════════════════════════════════════════════════════════════════
    # WINDOW HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _centre_window(self, w, h):
        sx = self.winfo_screenwidth()  // 2 - w // 2
        sy = self.winfo_screenheight() // 2 - h // 2
        self.geometry(f"{w}x{h}+{sx}+{sy}")

    # ══════════════════════════════════════════════════════════════════════════
    # MENU BAR
    # ══════════════════════════════════════════════════════════════════════════

    def _build_menu(self):
        menubar = tk.Menu(self, bg=COLOUR["bg_mid"],
                          fg=COLOUR["text_light"], activebackground=COLOUR["accent"])
        # File
        file_menu = tk.Menu(menubar, tearoff=0, bg=COLOUR["bg_mid"],
                             fg=COLOUR["text_light"])
        file_menu.add_command(label="Capture Image    Ctrl+S",
                               command=self._capture_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit             Alt+F4",
                               command=self._on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        # View
        view_menu = tk.Menu(menubar, tearoff=0, bg=COLOUR["bg_mid"],
                             fg=COLOUR["text_light"])
        view_menu.add_checkbutton(label="Show Contour Overlay",
                                   variable=self._show_contours)
        view_menu.add_command(label="Fullscreen  F11",
                               command=self._toggle_fullscreen)
        menubar.add_cascade(label="View", menu=view_menu)

        # Help
        help_menu = tk.Menu(menubar, tearoff=0, bg=COLOUR["bg_mid"],
                             fg=COLOUR["text_light"])
        help_menu.add_command(label="Keyboard Shortcuts",
                               command=self._show_shortcuts)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYOUT BUILDER
    # ══════════════════════════════════════════════════════════════════════════

    def _build_layout(self):
        # ── Title bar ─────────────────────────────────────────────────────────
        title_fr = tk.Frame(self, bg=COLOUR["bg_panel"], height=46)
        title_fr.pack(fill=tk.X, side=tk.TOP)
        tk.Label(title_fr, text="  🔬  Smart Endoscopic Assistance System",
                 font=("Segoe UI", 13, "bold"),
                 bg=COLOUR["bg_panel"], fg=COLOUR["accent2"]).pack(side=tk.LEFT, pady=6)
        self._clock_lbl = tk.Label(title_fr, text="",
                                    font=("Consolas", 10),
                                    bg=COLOUR["bg_panel"], fg=COLOUR["text_dim"])
        self._clock_lbl.pack(side=tk.RIGHT, padx=14)

        # ── Status bar (bottom) ───────────────────────────────────────────────
        status_fr = tk.Frame(self, bg=COLOUR["bg_mid"], height=24)
        status_fr.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_var = tk.StringVar(value="System Ready")
        tk.Label(status_fr, textvariable=self._status_var,
                 font=("Segoe UI", 9), bg=COLOUR["bg_mid"],
                 fg=COLOUR["green"], anchor="w").pack(side=tk.LEFT, padx=8)
        self._fps_lbl = tk.Label(status_fr, text="FPS: --",
                                  font=("Segoe UI", 9),
                                  bg=COLOUR["bg_mid"], fg=COLOUR["text_dim"])
        self._fps_lbl.pack(side=tk.RIGHT, padx=8)

        # ── Main body ─────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=COLOUR["bg_dark"])
        body.pack(fill=tk.BOTH, expand=True)

        # Left: video panels (2/3 width)
        video_fr = tk.Frame(body, bg=COLOUR["bg_dark"])
        video_fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_video_panels(video_fr)

        # Right: controls (1/3 width)
        ctrl_fr = tk.Frame(body, bg=COLOUR["bg_mid"], width=310)
        ctrl_fr.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_fr.pack_propagate(False)
        self._build_control_panel(ctrl_fr)

    # ── Video panels ──────────────────────────────────────────────────────────
    def _build_video_panels(self, parent):
        VW, VH = 620, 440
        PW, PH = 300, 220

        # Top row: Live | Processed
        top = tk.Frame(parent, bg=COLOUR["bg_dark"])
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 3))

        # Live view
        live_fr = tk.LabelFrame(top, text=" 📷 Live Endoscopic View ",
                                  font=("Segoe UI", 9, "bold"),
                                  bg=COLOUR["bg_dark"], fg=COLOUR["accent2"],
                                  bd=2, relief=tk.GROOVE)
        live_fr.pack(side=tk.LEFT, padx=(0, 4))
        self._live_canvas = tk.Canvas(live_fr, width=VW, height=VH,
                                       bg="black", highlightthickness=0)
        self._live_canvas.pack()
        self._live_img = None

        # Processed view
        proc_fr = tk.LabelFrame(top, text=" ⚗️  Processed View ",
                                  font=("Segoe UI", 9, "bold"),
                                  bg=COLOUR["bg_dark"], fg=COLOUR["yellow"],
                                  bd=2, relief=tk.GROOVE)
        proc_fr.pack(side=tk.LEFT)
        self._proc_canvas = tk.Canvas(proc_fr, width=PW, height=PH,
                                       bg="black", highlightthickness=0)
        self._proc_canvas.pack()
        self._proc_img = None

        # Bottom row: Feature panel | Capture preview
        bot = tk.Frame(parent, bg=COLOUR["bg_dark"])
        bot.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(3, 6))

        feat_fr = tk.LabelFrame(bot, text=" 📊 Feature Extraction ",
                                  font=("Segoe UI", 9, "bold"),
                                  bg=COLOUR["bg_dark"], fg=COLOUR["green"],
                                  bd=2, relief=tk.GROOVE)
        feat_fr.pack(side=tk.LEFT, padx=(0, 4))
        self._feat_canvas = tk.Canvas(feat_fr, width=PW, height=PH,
                                       bg="black", highlightthickness=0)
        self._feat_canvas.pack()
        self._feat_img = None

        cap_fr = tk.LabelFrame(bot, text=" 📸 Last Captured Image ",
                                 font=("Segoe UI", 9, "bold"),
                                 bg=COLOUR["bg_dark"], fg=COLOUR["red"],
                                 bd=2, relief=tk.GROOVE)
        cap_fr.pack(side=tk.LEFT)
        self._cap_canvas = tk.Canvas(cap_fr, width=PW, height=PH,
                                      bg="black", highlightthickness=0)
        self._cap_canvas.pack()
        self._cap_img = None

    # ── Control Panel ──────────────────────────────────────────────────────────
    def _build_control_panel(self, parent):
        style = {"font": ("Segoe UI", 9, "bold"),
                 "bg": COLOUR["bg_panel"],
                 "fg": COLOUR["text_light"],
                 "bd": 1, "relief": tk.GROOVE}

        def section(label, colour=COLOUR["accent2"]):
            fr = tk.LabelFrame(parent, text=f" {label} ",
                               font=("Segoe UI", 9, "bold"),
                               bg=COLOUR["bg_mid"], fg=colour,
                               bd=1, relief=tk.GROOVE)
            fr.pack(fill=tk.X, padx=8, pady=(6, 2))
            return fr

        # ── Illumination ───────────────────────────────────────────────────
        illum_fr = section("💡 Illumination", COLOUR["yellow"])
        self._brightness_var = tk.DoubleVar(value=70.0)
        tk.Label(illum_fr, text="Light Intensity",
                 **{**style, "bg": COLOUR["bg_mid"]}).pack(anchor="w", padx=6)
        self._bright_slider = tk.Scale(
            illum_fr, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self._brightness_var,
            command=self._on_brightness_change,
            bg=COLOUR["bg_mid"], fg=COLOUR["yellow"],
            troughcolor=COLOUR["bg_dark"], highlightthickness=0,
            length=280, resolution=1, showvalue=True,
            font=("Segoe UI", 8)
        )
        self._bright_slider.pack(padx=6, pady=2)
        self._bright_lbl = tk.Label(illum_fr, text="70 %",
                                     font=("Consolas", 9),
                                     bg=COLOUR["bg_mid"], fg=COLOUR["yellow"])
        self._bright_lbl.pack()

        # ── Camera controls ────────────────────────────────────────────────
        cam_fr = section("📷 Camera", COLOUR["accent2"])
        btn_row = tk.Frame(cam_fr, bg=COLOUR["bg_mid"])
        btn_row.pack(fill=tk.X, padx=6, pady=4)
        self._cam_btn = self._make_btn(btn_row, "■  Stop Camera",
                                        COLOUR["red"], self._toggle_camera,
                                        side=tk.LEFT)
        self._pause_btn = self._make_btn(btn_row, "⏸  Pause",
                                          COLOUR["yellow"], self._toggle_pause,
                                          side=tk.LEFT)

        # ── Capture ────────────────────────────────────────────────────────
        cap_fr2 = section("📸 Capture", COLOUR["green"])
        self._make_btn(cap_fr2, "📸  Capture Image  (Ctrl+S)",
                       COLOUR["green"], self._capture_image).pack(
            fill=tk.X, padx=6, pady=4)
        self._cap_status = tk.Label(cap_fr2, text="No capture yet",
                                     font=("Segoe UI", 8, "italic"),
                                     bg=COLOUR["bg_mid"], fg=COLOUR["text_dim"],
                                     wraplength=270)
        self._cap_status.pack(padx=6, pady=(0, 4))

        # ── Navigation ────────────────────────────────────────────────────
        nav_fr = section("🕹️ Navigation", COLOUR["accent"])
        self._nav_lbl = tk.Label(nav_fr, text="Stationary",
                                  font=("Segoe UI", 9, "bold"),
                                  bg=COLOUR["bg_mid"], fg=COLOUR["text_dim"])
        self._nav_lbl.pack(pady=2)
        self._build_nav_arrows(nav_fr)
        reset_row = tk.Frame(nav_fr, bg=COLOUR["bg_mid"])
        reset_row.pack(pady=4)
        self._make_btn(reset_row, "⟲  Reset Position",
                       COLOUR["accent"], self._reset_nav, side=tk.LEFT)
        zoom_fr = tk.Frame(nav_fr, bg=COLOUR["bg_mid"])
        zoom_fr.pack(fill=tk.X, padx=6, pady=2)
        self._make_btn(zoom_fr, "⊕ Forward",  COLOUR["accent2"],
                       lambda: self.nav.move("FORWARD"),  side=tk.LEFT)
        self._make_btn(zoom_fr, "⊖ Backward", COLOUR["red"],
                       lambda: self.nav.move("BACKWARD"), side=tk.LEFT)

        # ── Processing ────────────────────────────────────────────────────
        proc_fr = section("⚗️  Image Processing", COLOUR["text_light"])
        tk.Label(proc_fr, text="Noise Reduction",
                 **{**style, "bg": COLOUR["bg_mid"]}).pack(anchor="w", padx=6)
        self._noise_var = tk.StringVar(value="Bilateral")
        tk.OptionMenu(proc_fr, self._noise_var, *NOISE_METHODS,
                      command=self._on_proc_change).pack(fill=tk.X, padx=6, pady=2)
        tk.Label(proc_fr, text="Contrast Enhancement",
                 **{**style, "bg": COLOUR["bg_mid"]}).pack(anchor="w", padx=6)
        self._contrast_var = tk.StringVar(value="CLAHE")
        tk.OptionMenu(proc_fr, self._contrast_var, *CONTRAST_METHODS,
                      command=self._on_proc_change).pack(fill=tk.X, padx=6, pady=2)
        tk.Checkbutton(proc_fr, text="Show Contour Overlay",
                       variable=self._show_contours,
                       bg=COLOUR["bg_mid"], fg=COLOUR["text_light"],
                       selectcolor=COLOUR["bg_dark"],
                       activebackground=COLOUR["bg_mid"],
                       font=("Segoe UI", 9)).pack(anchor="w", padx=6, pady=2)

        # ── Exit ──────────────────────────────────────────────────────────
        sep = tk.Frame(parent, bg=COLOUR["accent"], height=1)
        sep.pack(fill=tk.X, padx=8, pady=8)
        self._make_btn(parent, "✕  EXIT", COLOUR["red"],
                       self._on_exit).pack(fill=tk.X, padx=8, pady=4)

    def _build_nav_arrows(self, parent):
        """3×3 directional button grid."""
        grid = tk.Frame(parent, bg=COLOUR["bg_mid"])
        grid.pack(pady=4)
        btn_cfg = {"font": ("Segoe UI", 11, "bold"), "width": 3, "height": 1,
                   "bg": COLOUR["bg_panel"], "fg": COLOUR["text_light"],
                   "activebackground": COLOUR["accent"],
                   "relief": tk.RAISED, "cursor": "hand2"}
        arrows = {
            (0, 1): ("↑", "UP"),
            (1, 0): ("←", "LEFT"),
            (1, 1): ("·", "NONE"),
            (1, 2): ("→", "RIGHT"),
            (2, 1): ("↓", "DOWN"),
        }
        for (r, c), (sym, d) in arrows.items():
            b = tk.Button(grid, text=sym, **btn_cfg)
            if d != "NONE":
                b.bind("<ButtonPress-1>",
                       lambda e, _d=d: self._nav_press(_d))
                b.bind("<ButtonRelease-1>",
                       lambda e: self._nav_release())
            b.grid(row=r, column=c, padx=2, pady=2)

    # ══════════════════════════════════════════════════════════════════════════
    # BUTTON / WIDGET FACTORY
    # ══════════════════════════════════════════════════════════════════════════

    def _make_btn(self, parent, text, colour, command, side=None):
        b = tk.Button(parent, text=text,
                      font=("Segoe UI", 9, "bold"),
                      bg=COLOUR["bg_panel"], fg=colour,
                      activebackground=colour, activeforeground="black",
                      relief=tk.RAISED, cursor="hand2",
                      command=command, padx=6, pady=4)
        if side is not None:
            b.pack(side=side, padx=3, pady=2)
        return b

    # ══════════════════════════════════════════════════════════════════════════
    # KEY BINDINGS
    # ══════════════════════════════════════════════════════════════════════════

    def _bind_keys(self):
        key_map = {
            "<Up>":    "UP",    "<Down>":  "DOWN",
            "<Left>":  "LEFT",  "<Right>": "RIGHT",
            "<Prior>": "FORWARD", "<Next>": "BACKWARD",  # PgUp / PgDn
        }
        for key, d in key_map.items():
            self.bind(key,
                      lambda e, _d=d: self._nav_press(_d))
            self.bind(f"<KeyRelease-{key[1:-1]}>",
                      lambda e: self._nav_release())

        self.bind("<Control-s>", lambda e: self._capture_image())
        self.bind("<space>",     lambda e: self._toggle_pause())
        self.bind("<Escape>",    lambda e: self.nav.reset())
        self.bind("<F11>",       lambda e: self._toggle_fullscreen())
        self.bind("<r>",         lambda e: self._reset_nav())

    # ══════════════════════════════════════════════════════════════════════════
    # RENDER LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def _update_loop(self):
        if not self._running:
            return
        try:
            self._tick()
        except Exception as exc:
            logger.error("Render loop error: %s", exc, exc_info=True)
        self.after(self.UPDATE_MS, self._update_loop)

    def _tick(self):
        # ── clock ─────────────────────────────────────────────────────────────
        self._clock_lbl.config(text=timestamp_display())

        # ── FPS ───────────────────────────────────────────────────────────────
        now = time.perf_counter()
        dt  = now - self._fps_ts
        if dt > 0:
            self._fps = 0.85 * self._fps + 0.15 * (1.0 / dt)
        self._fps_ts = now
        self._fps_lbl.config(text=f"FPS: {self._fps:.1f}")

        if not self._camera_on:
            return

        # ── grab frame ────────────────────────────────────────────────────────
        if not self._paused:
            ok, raw = self.camera.read()
            if ok:
                self._last_raw = raw
        else:
            raw = self._last_raw

        # ── illumination ──────────────────────────────────────────────────────
        lit = self.illum.apply(raw)

        # ── navigation ────────────────────────────────────────────────────────
        self.nav.apply_inertia()
        navigated = self.nav.apply(lit)

        # ── HUD overlay ───────────────────────────────────────────────────────
        hud_frame = draw_hud(navigated,
                              self.illum.brightness,
                              self.nav.label,
                              self._paused,
                              self._fps)
        hud_frame = self.nav.draw_hud(hud_frame)

        # ── optionally draw contours on live view ─────────────────────────────
        live_display = (draw_contours(hud_frame)
                        if self._show_contours.get() else hud_frame)

        # ── image processing (processed panel) ───────────────────────────────
        processed = self.processor.process(navigated)
        self._last_processed = processed

        # ── feature extraction (throttled) ────────────────────────────────────
        self._frame_count += 1
        if self._frame_count % self.FEATURE_PERIOD == 0:
            self._features = extract_all_features(processed)

        feat_panel = render_feature_panel(self._features, 300, 220)

        # ── update canvases ───────────────────────────────────────────────────
        self._update_canvas(self._live_canvas, live_display, 620, 440,
                             "_live_img")
        self._update_canvas(self._proc_canvas, processed, 300, 220,
                             "_proc_img")
        self._update_canvas(self._feat_canvas, feat_panel, 300, 220,
                             "_feat_img")
        self._update_canvas(self._cap_canvas, self._capture_preview, 300, 220,
                             "_cap_img")

        # ── navigation label ──────────────────────────────────────────────────
        self._nav_lbl.config(text=self.nav.label,
                              fg=(COLOUR["green"]
                                  if self.nav.current_direction != "NONE"
                                  else COLOUR["text_dim"]))

    def _update_canvas(self, canvas, frame, w, h, attr_name):
        ph = _to_photoimage(frame, w, h)
        canvas.create_image(0, 0, anchor=tk.NW, image=ph)
        setattr(self, attr_name, ph)   # prevent GC

    # ══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def _on_brightness_change(self, val):
        v = float(val)
        self.illum.set_brightness(v)
        self._bright_lbl.config(text=f"{int(v)} %")

    def _on_proc_change(self, *_):
        self.processor.noise_method    = self._noise_var.get()
        self.processor.contrast_method = self._contrast_var.get()
        self._set_status(f"Processing: {self.processor.noise_method} + "
                         f"{self.processor.contrast_method}")

    def _toggle_camera(self):
        self._camera_on = not self._camera_on
        if self._camera_on:
            self._cam_btn.config(text="■  Stop Camera",  fg=COLOUR["red"])
            self._set_status("Camera started.")
        else:
            self._cam_btn.config(text="▶  Start Camera", fg=COLOUR["green"])
            self._set_status("Camera stopped.")

    def _toggle_pause(self):
        self._paused = not self._paused
        self._pause_btn.config(
            text="▶  Resume" if self._paused else "⏸  Pause",
            fg=COLOUR["green"] if self._paused else COLOUR["yellow"]
        )
        self._set_status("Video paused." if self._paused else "Video resumed.")

    def _capture_image(self, *_):
        """
        Save both raw and processed frames; show comparison; update preview.
        """
        raw  = self._last_raw
        proc = self._last_processed
        if raw is None or raw.size == 0:
            messagebox.showwarning("Capture", "No frame available to capture.")
            return

        # Save raw
        path_raw  = save_image(raw,  prefix="raw_capture")
        path_proc = save_image(proc, prefix="proc_capture")

        # Build comparison image
        comparison = make_comparison(raw, proc,
                                      "Original", "Processed")
        path_comp  = save_image(comparison, prefix="comparison")

        # Update preview panel
        self._capture_preview = cv2.resize(proc, (PANEL_W, PANEL_H))

        # Update status
        filename = os.path.basename(path_raw)
        self._cap_status.config(text=f"✓ Saved: {filename}")
        self._set_status(f"Image captured → {path_raw}")

        # Popup report
        feat = self._features
        report = (
            f"Capture Summary\n"
            f"{'─'*40}\n"
            f"Raw image   : {os.path.basename(path_raw)}\n"
            f"Processed   : {os.path.basename(path_proc)}\n"
            f"Comparison  : {os.path.basename(path_comp)}\n"
            f"{'─'*40}\n"
            f"Brightness  : {self.illum.brightness:.0f}%\n"
            f"Zoom        : x{self.nav.zoom_level:.2f}\n"
            f"Noise filter: {self.processor.noise_method}\n"
            f"Contrast    : {self.processor.contrast_method}\n"
            f"{'─'*40}\n"
            f"Contours    : {feat.get('num_contours', '?')}\n"
            f"Circularity : {feat.get('circularity', '?')}\n"
            f"GLCM Energy : {feat.get('glcm_energy', '?')}\n"
            f"LBP Entropy : {feat.get('lbp_entropy', '?')}\n"
        )
        messagebox.showinfo("📸 Image Capture Report", report)

    def _nav_press(self, direction: str):
        self.nav.move(direction, continuous=True)

    def _nav_release(self):
        self.nav.stop_movement()

    def _reset_nav(self):
        self.nav.reset()
        self._set_status("Navigation reset.")

    def _toggle_fullscreen(self):
        current = self.attributes("-fullscreen")
        self.attributes("-fullscreen", not current)

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    # ══════════════════════════════════════════════════════════════════════════
    # HELP DIALOGS
    # ══════════════════════════════════════════════════════════════════════════

    def _show_shortcuts(self):
        shortcuts = (
            "Keyboard Shortcuts\n"
            "══════════════════\n"
            "Arrow Keys   : Navigate (Up/Down/Left/Right)\n"
            "PgUp / PgDn  : Forward / Backward (zoom)\n"
            "Space        : Pause / Resume video\n"
            "Ctrl + S     : Capture image\n"
            "Escape       : Reset navigation\n"
            "R            : Reset navigation\n"
            "F11          : Toggle fullscreen\n"
        )
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def _show_about(self):
        about = (
            "Smart Endoscopic Assistance System\n"
            "Medical Equipment II  ·  SBE3220\n\n"
            "Subsystems:\n"
            "   Illumination (LED simulation)\n"
            "   Imaging (synthetic/webcam/video)\n"
            "   Navigation (6-DOF simulation)\n"
            "   Image Processing (Bilateral + CLAHE)\n"
            "   Feature Extraction (Shape/Color/Texture)\n\n"
            "Built with Python, OpenCV, Tkinter, NumPy, scikit-image"
        )
        messagebox.showinfo("About", about)

    # ══════════════════════════════════════════════════════════════════════════
    # EXIT
    # ══════════════════════════════════════════════════════════════════════════

    def _on_exit(self):
        if messagebox.askokcancel("Exit", "Exit the Smart Endoscope System?"):
            self._running = False
            try:
                self.camera.stop()
            except Exception:
                pass
            self.destroy()
            logger.info("Application exited cleanly.")
