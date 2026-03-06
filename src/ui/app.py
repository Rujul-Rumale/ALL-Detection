"""
ALL Detection System - Medical AI Dashboard
Target Hardware: Raspberry Pi 5 + 1080p Official Monitor
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import math
import threading
import traceback
import sys
import os
import queue
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.stage1_screening import ALLScreener
from detection.blast_detector_v5 import detect_blasts
from detection.llm_utils import LLMGenerator
from ui.theme import THEME


# === CUSTOM WIDGETS ===

class ModernCard(ctk.CTkFrame):
    """Themed card. Callers can override any CTkFrame kwarg."""
    def __init__(self, parent, **kwargs):
        defaults = {
            "fg_color": THEME["surface"],
            "corner_radius": THEME["corner_radius"],
            "border_width": THEME["border_width"],
            "border_color": THEME["border_color"],
        }
        defaults.update(kwargs)
        super().__init__(parent, **defaults)


class CellCard(ModernCard):
    """Card for a single detected cell."""
    def __init__(self, parent, cell_data, crop_image=None, **kwargs):
        super().__init__(parent, **kwargs)

        is_blast = cell_data.get('is_blast', False)
        status_color = THEME["danger"] if is_blast else THEME["success"]

        if is_blast:
            self.configure(border_color=status_color, border_width=2)

        self.grid_columnconfigure(1, weight=1)

        # Thumbnail
        if crop_image is not None:
            pil_img = Image.fromarray(crop_image)
            pil_img = pil_img.resize((60, 60), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(pil_img, size=(60, 60))
            ctk.CTkLabel(self, image=ctk_img, text="").grid(
                row=0, column=0, rowspan=2, padx=12, pady=10)

        # Header
        header_text = f"Cell #{cell_data['id']} • {cell_data['classification']}"
        ctk.CTkLabel(
            self, text=header_text,
            font=THEME["font_main"], text_color=status_color
        ).grid(row=0, column=1, sticky="sw", padx=(0, 10), pady=(10, 2))

        # Metrics
        confidence = cell_data.get('confidence', cell_data.get('score', 0))
        metrics = f"Circ: {cell_data.get('circularity', 0)*100:.0f}%  |  TFLite Conf: {confidence*100:.1f}%"
        ctk.CTkLabel(
            self, text=metrics,
            font=THEME["font_sm"], text_color=THEME["text_dim"]
        ).grid(row=1, column=1, sticky="nw", padx=(0, 10), pady=(0, 10))


class ShimmerDots(tk.Canvas):
    """
    Premium thinking animation — 3 dots that pulse in sequence
    with smooth sinusoidal brightness, similar to Gemini/Claude.
    """
    DOT_COUNT = 3
    DOT_RADIUS = 5
    DOT_SPACING = 22
    SPEED = 60  # ms per frame

    def __init__(self, parent, **kwargs):
        # Parse accent color to RGB
        self._accent = THEME["accent"]
        self._bg = THEME["surface"]
        super().__init__(
            parent,
            width=self.DOT_COUNT * self.DOT_SPACING + 10,
            height=self.DOT_RADIUS * 2 + 10,
            bg=self._bg, highlightthickness=0, **kwargs
        )
        self._running = True
        self._tick = 0
        self._dots = []

        cy = self.DOT_RADIUS + 5
        for i in range(self.DOT_COUNT):
            cx = self.DOT_RADIUS + 5 + i * self.DOT_SPACING
            d = self.create_oval(
                cx - self.DOT_RADIUS, cy - self.DOT_RADIUS,
                cx + self.DOT_RADIUS, cy + self.DOT_RADIUS,
                fill=self._accent, outline=""
            )
            self._dots.append(d)

        self._animate()

    def _animate(self):
        if not self._running:
            return
        self._tick += 1
        for i, dot in enumerate(self._dots):
            # Each dot offset by phase → wave effect
            phase = (self._tick * 0.15) - (i * 1.2)
            brightness = 0.35 + 0.65 * max(0, math.sin(phase))
            color = self._blend(self._bg, self._accent, brightness)
            # Scale dot
            scale = 0.7 + 0.3 * max(0, math.sin(phase))
            cy = self.DOT_RADIUS + 5
            cx = self.DOT_RADIUS + 5 + i * self.DOT_SPACING
            r = self.DOT_RADIUS * scale
            self.coords(dot, cx - r, cy - r, cx + r, cy + r)
            self.itemconfig(dot, fill=color)
        self.after(self.SPEED, self._animate)

    def _blend(self, bg_hex, fg_hex, t):
        """Blend two hex colors by factor t."""
        bg = self._hex_to_rgb(bg_hex)
        fg = self._hex_to_rgb(fg_hex)
        r = int(bg[0] + (fg[0] - bg[0]) * t)
        g = int(bg[1] + (fg[1] - bg[1]) * t)
        b = int(bg[2] + (fg[2] - bg[2]) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _hex_to_rgb(self, h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def fade_out(self, callback, steps=10, step=0):
        """Gradually fade dots to background and shrink, then call callback."""
        if step >= steps:
            callback()
            return
        t = step / steps
        for dot in self._dots:
            color = self._blend(self._accent, self._bg, t)
            self.itemconfig(dot, fill=color)
        self.after(50, lambda: self.fade_out(callback, steps, step + 1))

    def stop(self):
        self._running = False


class AIStatusWidget(ctk.CTkFrame):
    """
    Premium AI thinking indicator with shimmer dots.
    Smooth transition to 'Analysis Complete' when done.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._running = True

        # Top row: label + shimmer dots
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x")

        self.status_label = ctk.CTkLabel(
            row, text="Analyzing with Phi-3",
            font=("Roboto", 13, "bold"), text_color=THEME["accent"]
        )
        self.status_label.pack(side="left")

        self.shimmer = ShimmerDots(row)
        self.shimmer.pack(side="left", padx=(8, 0))

    def set_complete(self):
        """Smooth transition: fade dots → show checkmark."""
        self.shimmer.stop()
        self.shimmer.fade_out(self._show_complete)

    def _show_complete(self):
        """Final state after fade."""
        self.shimmer.pack_forget()
        self.status_label.configure(
            text="✓  AI Analysis Complete",
            text_color=THEME["success"]
        )


class AITextBox(ctk.CTkTextbox):
    """Scrollable text area that accepts streamed tokens. Stays inside its frame."""
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            font=("Roboto", 14),
            text_color=THEME["text"],
            fg_color="transparent",
            wrap="word",
            activate_scrollbars=False,
            state="disabled",
            **kwargs
        )

    def append_text(self, chunk):
        self.configure(state="normal")
        self.insert("end", chunk)
        self.see("end")
        self.configure(state="disabled")


# === MAIN APPLICATION ===

class ALLDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window ---
        self.title("ALL Detection System")
        self.configure(fg_color=THEME["bg"])

        # Proportional sizing: 75% of screen, minimum 1024x600
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = max(int(screen_w * 0.75), 1024)
        win_h = max(int(screen_h * 0.75), 600)
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(900, 500)

        # --- State ---
        self.screener = None
        self.llm_queue = queue.Queue()
        self.ollama_process = None
        self.ai_status_widget = None

        # --- Build UI immediately (fast) ---
        self._build_ui()

        # --- Deferred heavy work (non-blocking) ---
        self.after(100, self._process_ui_queue)
        threading.Thread(target=self._boot_services, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    #  Layout — fixed proportional panels                                 #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        """
        Fixed proportional layout using grid with uniform groups.
        Panels have fixed proportions that don't change with content.
        Adapts to any screen size via weights.
        """

        # ─── Top Bar ─────────────────────────────────────────────
        top = ctk.CTkFrame(self, height=56, fg_color=THEME["surface"], corner_radius=0)
        top.pack(fill="x")
        top.pack_propagate(False)

        ctk.CTkLabel(
            top, text="ALL Detection System",
            font=THEME["font_hero"], text_color=THEME["primary"]
        ).pack(side="left", padx=25)

        self.status_lbl = ctk.CTkLabel(
            top, text="Booting...",
            font=THEME["font_sm"], text_color=THEME["text_dim"]
        )
        self.status_lbl.pack(side="right", padx=25)

        self.btn_load = ctk.CTkButton(
            top, text="📂  Load Sample", command=self._load_image,
            height=36, corner_radius=10,
            font=THEME["font_main"],
            fg_color=THEME["primary"], text_color=THEME["bg"],
            hover_color=THEME["primary_hover"]
        )
        self.btn_load.pack(side="right", padx=10, pady=10)

        # ─── Body Container ──────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=(12, 8))

        # Uniform groups lock all columns/rows in a group to equal size
        # Left = 2 units, Right = 3 units → 40% / 60%
        body.grid_columnconfigure(0, weight=2, uniform="cols")
        body.grid_columnconfigure(1, weight=3, uniform="cols")
        # Top = 3 units, Bottom = 2 units → 60% / 40%
        body.grid_rowconfigure(0, weight=3, uniform="rows")
        body.grid_rowconfigure(1, weight=2, uniform="rows")

        # ─── Left Top: Image Panel ───────────────────────────────
        self.image_frame = ctk.CTkFrame(body, fg_color=THEME["surface"], corner_radius=THEME["corner_radius"])
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))
        self.image_frame.pack_propagate(False)

        self.img_label = ctk.CTkLabel(
            self.image_frame, text="Load an image to begin analysis",
            font=THEME["font_header"], text_color=THEME["text_dim"]
        )
        self.img_label.pack(expand=True, fill="both", padx=10, pady=10)

        # ─── Left Bottom: AI Summary Panel ───────────────────────
        self.ai_frame = ctk.CTkFrame(body, fg_color=THEME["surface"], corner_radius=THEME["corner_radius"])
        self.ai_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(5, 0))
        self.ai_frame.pack_propagate(False)

        self.ai_placeholder = ctk.CTkLabel(
            self.ai_frame, text="AI summary will appear here",
            font=THEME["font_sm"], text_color=THEME["text_dim"]
        )
        self.ai_placeholder.pack(expand=True, pady=15)

        # ─── Right: Results Panel (full height) ──────────────────
        results_panel = ctk.CTkFrame(body, fg_color=THEME["surface"], corner_radius=THEME["corner_radius"])
        results_panel.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(5, 0))

        # Banner
        self.result_banner = ctk.CTkLabel(
            results_panel, text="No analysis yet",
            font=THEME["font_header"], text_color=THEME["text_dim"], height=50
        )
        self.result_banner.pack(fill="x", padx=15, pady=(15, 5))

        # Scrollable cell list
        self.feed_scroll = ctk.CTkScrollableFrame(
            results_panel, fg_color="transparent", label_text=""
        )
        self.feed_scroll.pack(fill="both", expand=True, padx=10, pady=(5, 15))

        # ─── Bottom Status Bar ────────────────────────────────────
        bottom = ctk.CTkFrame(self, height=28, fg_color=THEME["surface"], corner_radius=0)
        bottom.pack(fill="x")
        bottom.pack_propagate(False)

        self.progress_lbl = ctk.CTkLabel(
            bottom, text="", font=THEME["font_sm"], text_color=THEME["text_dim"]
        )
        self.progress_lbl.pack(side="left", padx=15)

    # ------------------------------------------------------------------ #
    #  Background Services                                                #
    # ------------------------------------------------------------------ #
    def _boot_services(self):
        """Load models + start Ollama — all in one background thread."""
        try:
            self.screener = ALLScreener()
            self._safe_status("Models loaded ✓")
        except Exception as e:
            print(f"Model Init: {e}")
            self._safe_status("Model load failed")

        self._manage_ollama()

    def _manage_ollama(self):
        """Check / start Ollama service (runs in background thread)."""
        try:
            if LLMGenerator._check_connection("phi3"):
                self._safe_status("System Ready  •  AI: Linked ✓")
                return

            self._safe_status("Starting AI service...")
            try:
                flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=flags
                )
                import time
                for _ in range(4):
                    time.sleep(2)
                    if LLMGenerator._check_connection("phi3"):
                        self._safe_status("System Ready  •  AI: Started ✓")
                        return
                self._safe_status("System Ready  •  AI: Timeout ⚠")
            except FileNotFoundError:
                self._safe_status("System Ready  •  AI: Not Installed")
        except Exception:
            self._safe_status("System Ready  •  AI: Error")

    def _on_close(self):
        if self.ollama_process:
            print("Stopping Ollama service...")
            self.ollama_process.terminate()
        self.destroy()

    # ------------------------------------------------------------------ #
    #  UI Queue (thread-safe LLM token delivery)                          #
    # ------------------------------------------------------------------ #
    def _process_ui_queue(self):
        try:
            while True:
                msg_type, data = self.llm_queue.get_nowait()
                if msg_type == "token" and hasattr(self, 'ai_textbox'):
                    self.ai_textbox.append_text(data)
                elif msg_type == "done":
                    if self.ai_status_widget:
                        self.ai_status_widget.set_complete()
        except queue.Empty:
            pass
        finally:
            self.after(50, self._process_ui_queue)

    # ------------------------------------------------------------------ #
    #  Analysis Pipeline                                                  #
    # ------------------------------------------------------------------ #
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select Blood Smear",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All", "*.*")]
        )
        if path:
            self._start_analysis(path)

    def _start_analysis(self, path):
        # Clear previous results
        for w in self.feed_scroll.winfo_children():
            w.destroy()
        for w in self.ai_frame.winfo_children():
            w.destroy()
        self.ai_status_widget = None

        self._display_image(path)
        self.result_banner.configure(text="Processing...", text_color=THEME["warning"])
        self.btn_load.configure(state="disabled")
        threading.Thread(target=self._run_pipeline, args=(path,), daemon=True).start()

    def _run_pipeline(self, path):
        try:
            self._safe_progress("Stage 1: Detecting cells...")
            detection = detect_blasts(path, return_crops=True, return_all_cells=True)

            # Update image with bounding boxes
            if detection.get('annotated_image') is not None:
                self.after(0, lambda: self._display_array(detection['annotated_image']))

            # Populate results feed (all at once — no stagger = no tearing)
            self.after(0, lambda: self._populate_results(detection))

            # AI summary
            blasts = [c for c in detection['detections'] if c.get('is_blast')]
            if blasts:
                self.after(0, self._show_ai_thinking)
                for token in LLMGenerator.generate_explanation_stream(blasts):
                    self.llm_queue.put(("token", token))
                self.llm_queue.put(("done", None))

            self._safe_progress("Analysis complete")
        except Exception as e:
            traceback.print_exc()
            self._safe_progress(f"Error: {str(e)[:40]}")
        finally:
            self.after(0, lambda: self.btn_load.configure(state="normal"))

    # ------------------------------------------------------------------ #
    #  Results Rendering                                                  #
    # ------------------------------------------------------------------ #
    def _populate_results(self, detection):
        blast_cnt = detection['blast_count']
        total = detection['total_cells']

        # Banner
        if blast_cnt > 0:
            self.result_banner.configure(
                text=f"⚠  SUSPECTED ALL — {blast_cnt} Blast(s) / {total} Cells",
                text_color=THEME["danger"]
            )
        else:
            self.result_banner.configure(
                text=f"✅  HEALTHY — {total} Cells Analyzed",
                text_color=THEME["success"]
            )

        # Add all cards at once (no stagger = no tearing)
        cells = sorted(detection['detections'], key=lambda x: not x.get('is_blast', False))
        for cell in cells:
            CellCard(self.feed_scroll, cell, crop_image=cell.get('crop')).pack(
                fill="x", padx=8, pady=4)

    def _show_ai_thinking(self):
        """Set up AI frame with shimmer + scrollable text area."""
        for w in self.ai_frame.winfo_children():
            w.destroy()

        # Shimmer dots status
        self.ai_status_widget = AIStatusWidget(self.ai_frame)
        self.ai_status_widget.pack(fill="x", padx=15, pady=(12, 6))

        # Thin separator
        ctk.CTkFrame(self.ai_frame, height=1, fg_color=THEME["border_color"]).pack(
            fill="x", padx=15)

        # Scrollable text area (properly contained)
        self.ai_textbox = AITextBox(self.ai_frame)
        self.ai_textbox.pack(fill="both", expand=True, padx=10, pady=(6, 10))

    # ------------------------------------------------------------------ #
    #  Image Display                                                      #
    # ------------------------------------------------------------------ #
    def _display_image(self, path):
        try:
            self._display_pil(Image.open(path))
        except Exception:
            pass

    def _display_array(self, arr):
        try:
            self._display_pil(Image.fromarray(arr))
        except Exception:
            pass

    def _display_pil(self, pil_img):
        self.image_frame.update_idletasks()
        max_w = max(self.image_frame.winfo_width() - 20, 200)
        max_h = max(self.image_frame.winfo_height() - 20, 200)

        ratio = min(max_w / pil_img.width, max_h / pil_img.height)
        new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        ctk_img = ctk.CTkImage(pil_img, size=new_size)
        self.img_label.configure(image=ctk_img, text="")
        self.img_label.image = ctk_img

    # ------------------------------------------------------------------ #
    #  Thread-safe helpers                                                #
    # ------------------------------------------------------------------ #
    def _safe_status(self, text):
        self.after(0, lambda: self.status_lbl.configure(text=text))

    def _safe_progress(self, text):
        self.after(0, lambda: self.progress_lbl.configure(text=text))


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    app = ALLDetectionApp()
    app.mainloop()
