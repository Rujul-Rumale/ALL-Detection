"""
ALL Detection System — Refreshed 3-Panel Classification UI
Layout: Image + Contours | Cell Grid | Analytics Panel
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import threading
import math
import traceback
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.theme import THEME
from detection.demo_pipeline import DemoPipeline, MODELS


# ═══════════════════════════════════════════
# CELL THUMBNAIL WIDGET
# ═══════════════════════════════════════════
class CellThumbnail(ctk.CTkFrame):
    """Compact cell crop with colored border, ID and confidence."""
    
    THUMB_SIZE = 100
    
    def __init__(self, parent, cell_data, **kwargs):
        is_blast = cell_data.get('is_blast', False)
        is_debris = cell_data.get('is_debris', False)
        
        if is_debris:
            border_color = THEME["debris_border"]
        elif is_blast:
            border_color = THEME["blast_border"]
        else:
            border_color = THEME["healthy_border"]
        
        super().__init__(
            parent,
            fg_color=THEME["surface"],
            corner_radius=10,
            border_width=2,
            border_color=border_color,
            **kwargs
        )
        
        # Cell image
        crop = cell_data.get('crop')
        if crop is not None:
            pil_img = Image.fromarray(crop)
            pil_img = pil_img.resize((self.THUMB_SIZE, self.THUMB_SIZE), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(pil_img, size=(self.THUMB_SIZE, self.THUMB_SIZE))
            img_label = ctk.CTkLabel(self, image=ctk_img, text="")
            img_label.image = ctk_img
            img_label.pack(padx=4, pady=(4, 0))
        
        # Classification label
        if is_debris:
            reason = cell_data.get('debris_reason', 'unknown')
            label_text = f"#{cell_data.get('id', '?')}  DEBRIS"
            label_color = THEME["debris_border"]
        else:
            class_name = cell_data.get('classification', '?')
            conf = cell_data.get('confidence', 0)
            label_text = f"#{cell_data.get('id', '?')}  {class_name}  {conf:.0%}"
            label_color = THEME["blast_border"] if is_blast else THEME["healthy_border"]
        
        ctk.CTkLabel(
            self,
            text=label_text,
            font=THEME["font_xs"],
            text_color=label_color,
        ).pack(pady=(2, 4))


# ═══════════════════════════════════════════
# DONUT CHART (Canvas-drawn)
# ═══════════════════════════════════════════
class DonutChart(ctk.CTkFrame):
    """Simple donut chart for blast vs healthy ratio."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.canvas = tk.Canvas(
            self, width=180, height=180,
            bg=THEME["surface"], highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=5)
        self.draw_empty()
    
    def draw_empty(self):
        self.canvas.delete("all")
        cx, cy, r = 90, 90, 70
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=THEME["chart_bg"], outline=THEME["border_color"], width=1
        )
        inner_r = 45
        self.canvas.create_oval(
            cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
            fill=THEME["surface"], outline=""
        )
        self.canvas.create_text(
            cx, cy, text="—", fill=THEME["text_dim"], font=("Roboto", 14)
        )
    
    def update_chart(self, blast_count, healthy_count):
        self.canvas.delete("all")
        total = blast_count + healthy_count
        cx, cy, r = 90, 90, 70
        inner_r = 45
        
        if total == 0:
            self.draw_empty()
            return
        
        blast_ratio = blast_count / total
        healthy_ratio = healthy_count / total
        
        # Draw arcs
        start = 90  # Start at top
        if blast_count > 0:
            extent = -blast_ratio * 360
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=start, extent=extent,
                fill=THEME["blast_border"], outline=""
            )
            start += extent
        
        if healthy_count > 0:
            extent = -healthy_ratio * 360
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=start, extent=extent,
                fill=THEME["healthy_border"], outline=""
            )
        
        # Inner circle (donut hole)
        self.canvas.create_oval(
            cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
            fill=THEME["surface"], outline=""
        )
        
        # Center text
        pct = f"{blast_ratio:.0%}" if blast_count > 0 else "0%"
        self.canvas.create_text(
            cx, cy - 8, text=pct, fill=THEME["text"], font=("Roboto", 18, "bold")
        )
        self.canvas.create_text(
            cx, cy + 12, text="blast", fill=THEME["text_dim"], font=("Roboto", 10)
        )


# ═══════════════════════════════════════════
# STAT PILL WIDGET
# ═══════════════════════════════════════════
class StatPill(ctk.CTkFrame):
    """Small stat display: label + value."""
    
    def __init__(self, parent, label, value="—", value_color=None, **kwargs):
        super().__init__(parent, fg_color=THEME["chart_bg"], corner_radius=8, **kwargs)
        
        ctk.CTkLabel(
            self, text=label,
            font=THEME["font_xs"], text_color=THEME["text_dim"]
        ).pack(pady=(6, 0))
        
        self.value_label = ctk.CTkLabel(
            self, text=str(value),
            font=("Roboto", 16, "bold"),
            text_color=value_color or THEME["text"]
        )
        self.value_label.pack(pady=(0, 6))
    
    def set_value(self, value, color=None):
        self.value_label.configure(text=str(value))
        if color:
            self.value_label.configure(text_color=color)


# ═══════════════════════════════════════════
# TIMING BAR WIDGET
# ═══════════════════════════════════════════
class TimingBar(ctk.CTkFrame):
    """Displays pipeline stage timing as a stacked horizontal bar."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.canvas = tk.Canvas(
            self, height=30, bg=THEME["surface"], highlightthickness=0
        )
        self.canvas.pack(fill="x", padx=5, pady=5)
        
        # Legend below
        self.legend_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.legend_frame.pack(fill="x", padx=5)
    
    def update_timing(self, watershed_t, sam_t, classify_t):
        self.canvas.delete("all")
        for w in self.legend_frame.winfo_children():
            w.destroy()
        
        total = watershed_t + sam_t + classify_t
        if total == 0:
            return
        
        self.canvas.update_idletasks()
        bar_w = max(self.canvas.winfo_width() - 10, 200)
        bar_h = 18
        x = 5
        y = 6
        
        colors = ["#4CC9F0", "#7209B7", "#FFAA00"]
        labels = [
            f"Watershed {watershed_t:.1f}s",
            f"SAM {sam_t:.1f}s",
            f"Classify {classify_t:.1f}s"
        ]
        times = [watershed_t, sam_t, classify_t]
        
        for i, t in enumerate(times):
            w = max(int((t / total) * bar_w), 2)
            self.canvas.create_rectangle(
                x, y, x + w, y + bar_h,
                fill=colors[i], outline=""
            )
            x += w
        
        # Legend
        for i, label in enumerate(labels):
            pill = ctk.CTkFrame(self.legend_frame, fg_color="transparent")
            pill.pack(side="left", padx=8)
            ctk.CTkLabel(pill, text="●", text_color=colors[i], font=("Roboto", 8)).pack(side="left")
            ctk.CTkLabel(pill, text=label, font=THEME["font_xs"], text_color=THEME["text_dim"]).pack(side="left", padx=(2, 0))


# ═══════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════
class ClassificationDemoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window ---
        self.title("ALL Detection System")
        self.configure(fg_color=THEME["bg"])

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = max(int(screen_w * 0.85), 1200)
        win_h = max(int(screen_h * 0.80), 700)
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(1100, 600)

        # --- State ---
        self.pipeline = DemoPipeline()
        self.selected_model = ctk.StringVar(value=list(MODELS.keys())[0])

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # ─── Header Bar ──────────────────────────────────────
        header = ctk.CTkFrame(self, height=60, fg_color=THEME["surface"], corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🔬  ALL Detection System",
            font=THEME["font_hero"], text_color=THEME["primary"]
        ).pack(side="left", padx=20)

        # Load button
        self.btn_load = ctk.CTkButton(
            header, text="📂  Load Image", command=self._load_image,
            height=36, corner_radius=10, width=140,
            font=THEME["font_main"],
            fg_color=THEME["primary"], text_color=THEME["bg"],
            hover_color=THEME["primary_hover"]
        )
        self.btn_load.pack(side="right", padx=20, pady=12)

        # Model selector
        model_frame = ctk.CTkFrame(header, fg_color="transparent")
        model_frame.pack(side="right", padx=10, pady=8)
        ctk.CTkLabel(
            model_frame, text="Model:",
            font=THEME["font_xs"], text_color=THEME["text_dim"]
        ).pack(side="left", padx=(0, 5))
        self.model_dropdown = ctk.CTkOptionMenu(
            model_frame,
            values=list(MODELS.keys()),
            variable=self.selected_model,
            font=THEME["font_sm"],
            width=220,
            fg_color=THEME["bg"], button_color=THEME["bg"],
            dropdown_hover_color=THEME["primary"]
        )
        self.model_dropdown.pack(side="left")

        # ─── 3-Column Body ────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        body.grid_columnconfigure(0, weight=3, uniform="main")
        body.grid_columnconfigure(1, weight=4, uniform="main")
        body.grid_columnconfigure(2, weight=3, uniform="main")
        body.grid_rowconfigure(0, weight=1)

        # ─── LEFT: Image Panel ─────────────────────────────
        left = ctk.CTkFrame(body, fg_color=THEME["surface"], corner_radius=THEME["corner_radius"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        ctk.CTkLabel(
            left, text="Blood Smear",
            font=THEME["font_sm"], text_color=THEME["text_dim"]
        ).pack(pady=(8, 0))

        self.img_label = ctk.CTkLabel(
            left, text="Load an image to begin\nWatershed → SAM → Classify",
            font=THEME["font_main"], text_color=THEME["text_dim"]
        )
        self.img_label.pack(expand=True, fill="both", padx=8, pady=4)

        # Stats bar below image
        stats_bar = ctk.CTkFrame(left, fg_color=THEME["chart_bg"], corner_radius=10, height=60)
        stats_bar.pack(fill="x", padx=8, pady=(0, 8))
        stats_bar.pack_propagate(False)

        stats_bar.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.stat_total = StatPill(stats_bar, "Total")
        self.stat_total.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        
        self.stat_blast = StatPill(stats_bar, "Blasts", value_color=THEME["blast_border"])
        self.stat_blast.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        
        self.stat_healthy = StatPill(stats_bar, "Healthy", value_color=THEME["healthy_border"])
        self.stat_healthy.grid(row=0, column=2, sticky="nsew", padx=3, pady=3)
        
        self.stat_ratio = StatPill(stats_bar, "Blast %")
        self.stat_ratio.grid(row=0, column=3, sticky="nsew", padx=3, pady=3)

        # ─── CENTER: Cell Grid ────────────────────────────
        center = ctk.CTkFrame(body, fg_color=THEME["surface"], corner_radius=THEME["corner_radius"])
        center.grid(row=0, column=1, sticky="nsew", padx=4)

        ctk.CTkLabel(
            center, text="Detected Cells",
            font=THEME["font_sm"], text_color=THEME["text_dim"]
        ).pack(pady=(8, 0))

        self.cell_grid_scroll = ctk.CTkScrollableFrame(
            center, fg_color="transparent"
        )
        self.cell_grid_scroll.pack(fill="both", expand=True, padx=6, pady=(4, 8))

        # Grid config: 4 columns
        self.grid_columns = 4

        self.grid_placeholder = ctk.CTkLabel(
            self.cell_grid_scroll, 
            text="Cell thumbnails will appear here\nafter analysis",
            font=THEME["font_main"], text_color=THEME["text_dim"]
        )
        self.grid_placeholder.pack(expand=True, pady=40)

        # ─── RIGHT: Analytics Panel ────────────────────────
        right = ctk.CTkFrame(body, fg_color=THEME["surface"], corner_radius=THEME["corner_radius"])
        right.grid(row=0, column=2, sticky="nsew", padx=(4, 0))

        ctk.CTkLabel(
            right, text="Analytics",
            font=THEME["font_sm"], text_color=THEME["text_dim"]
        ).pack(pady=(8, 0))

        # Verdict banner
        self.verdict_label = ctk.CTkLabel(
            right, text="Awaiting analysis",
            font=("Roboto", 16, "bold"), text_color=THEME["text_dim"],
            height=40
        )
        self.verdict_label.pack(fill="x", padx=12, pady=(8, 0))

        # Donut chart
        self.donut = DonutChart(right)
        self.donut.pack(pady=(4, 0))

        # Timing card
        timing_card = ctk.CTkFrame(right, fg_color=THEME["chart_bg"], corner_radius=10)
        timing_card.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            timing_card, text="Pipeline Timing",
            font=THEME["font_xs"], text_color=THEME["text_dim"]
        ).pack(anchor="w", padx=10, pady=(6, 0))

        self.timing_bar = TimingBar(timing_card)
        self.timing_bar.pack(fill="x", padx=4, pady=(0, 6))

        # Classification summary
        summary_card = ctk.CTkFrame(right, fg_color=THEME["chart_bg"], corner_radius=10)
        summary_card.pack(fill="x", padx=10, pady=(4, 10), expand=True)

        ctk.CTkLabel(
            summary_card, text="Classification Summary",
            font=THEME["font_xs"], text_color=THEME["text_dim"]
        ).pack(anchor="w", padx=10, pady=(6, 0))

        self.summary_text = ctk.CTkTextbox(
            summary_card, font=THEME["font_sm"],
            text_color=THEME["text"], fg_color="transparent",
            wrap="word", activate_scrollbars=False, state="disabled"
        )
        self.summary_text.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        # ─── Bottom Status Bar ───────────────────────────────
        bottom = ctk.CTkFrame(self, height=28, fg_color=THEME["surface"], corner_radius=0)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        self.progress_lbl = ctk.CTkLabel(
            bottom, text="Ready — load a blood smear image to begin",
            font=THEME["font_xs"], text_color=THEME["text_dim"]
        )
        self.progress_lbl.pack(side="left", padx=15)

    # ─── ACTIONS ─────────────────────────────────────────
    def _on_close(self):
        self.destroy()

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select Blood Smear",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All", "*.*")]
        )
        if path:
            self._start_analysis(path)

    def _start_analysis(self, path):
        # Clear previous
        for w in self.cell_grid_scroll.winfo_children():
            w.destroy()

        self._display_image(path)
        self.verdict_label.configure(text="Analyzing...", text_color=THEME["warning"])
        self.btn_load.configure(state="disabled")
        self.model_dropdown.configure(state="disabled")

        # Reset stats
        self.stat_total.set_value("…")
        self.stat_blast.set_value("…")
        self.stat_healthy.set_value("…")
        self.stat_ratio.set_value("…")
        self.donut.draw_empty()

        threading.Thread(target=self._run_pipeline, args=(path,), daemon=True).start()

    def _run_pipeline(self, path):
        try:
            self._safe_progress("Stage 1: Watershed centroid detection...")
            model_name = self.selected_model.get()
            results = self.pipeline.process_image(path, model_name)

            # Update image with contour overlays
            if results.get('annotated_image') is not None:
                self.after(0, lambda: self._display_array(results['annotated_image']))

            # Populate all panels
            self.after(0, lambda: self._populate_results(results))
            total_t = results.get('total_time', 0)
            total_cells = results.get('total_cells', 0)
            self._safe_progress(f"Analysis complete — {total_cells} cells detected in {total_t:.1f}s")

        except Exception as e:
            traceback.print_exc()
            self._safe_progress(f"Error: {str(e)[:60]}")
            self.after(0, lambda: self.verdict_label.configure(
                text="Analysis failed", text_color=THEME["danger"]))
        finally:
            self.after(0, lambda: self.btn_load.configure(state="normal"))
            self.after(0, lambda: self.model_dropdown.configure(state="normal"))

    def _populate_results(self, results):
        blast_cnt = results.get('blast_count', 0)
        healthy_cnt = results.get('healthy_count', 0)
        total = results.get('total_cells', 0)
        debris_cnt = results.get('debris_count', 0)

        # ── Stats bar ──
        self.stat_total.set_value(total)
        self.stat_blast.set_value(blast_cnt, THEME["blast_border"])
        self.stat_healthy.set_value(healthy_cnt, THEME["healthy_border"])
        
        ratio = f"{(blast_cnt / total * 100):.0f}%" if total > 0 else "—"
        ratio_color = THEME["blast_border"] if blast_cnt > 0 else THEME["healthy_border"]
        self.stat_ratio.set_value(ratio, ratio_color)

        # ── Verdict ──
        if total == 0:
            self.verdict_label.configure(
                text="No cells detected", text_color=THEME["text_dim"])
        elif blast_cnt > 0:
            self.verdict_label.configure(
                text=f"⚠  SUSPECTED ALL", text_color=THEME["danger"])
        else:
            self.verdict_label.configure(
                text="✅  HEALTHY", text_color=THEME["success"])

        # ── Donut chart ──
        self.donut.update_chart(blast_cnt, healthy_cnt)

        # ── Timing bar ──
        self.timing_bar.update_timing(
            results.get('watershed_time', 0),
            results.get('sam_time', 0),
            results.get('classify_time', 0)
        )

        # ── Classification summary text ──
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        
        summary_lines = [
            f"Model: {self.selected_model.get()}",
            f"Total Cells: {total}",
            f"Blast Cells (ALL): {blast_cnt}",
            f"Healthy Cells (HEM): {healthy_cnt}",
            f"Blast Ratio: {ratio}",
            f"Debris Detected: {debris_cnt}",
            "",
        ]
        
        # Per-cell breakdown
        detections = results.get('detections', [])
        cells_only = [d for d in detections if not d.get('is_debris', False)]
        debris_only = [d for d in detections if d.get('is_debris', False)]
        
        if cells_only:
            high_conf = [d for d in cells_only if d['confidence'] >= 0.9]
            med_conf = [d for d in cells_only if 0.7 <= d['confidence'] < 0.9]
            low_conf = [d for d in cells_only if d['confidence'] < 0.7]
            
            summary_lines.append("Confidence Distribution:")
            summary_lines.append(f"  High (≥90%): {len(high_conf)} cells")
            summary_lines.append(f"  Medium (70-90%): {len(med_conf)} cells")
            summary_lines.append(f"  Low (<70%): {len(low_conf)} cells")
        
        if debris_only:
            summary_lines.append("")
            summary_lines.append("Debris Breakdown:")
            reasons = {}
            for d in debris_only:
                r = d.get('debris_reason', 'unknown')
                reasons[r] = reasons.get(r, 0) + 1
            for reason, count in reasons.items():
                summary_lines.append(f"  {reason}: {count}")
        
        self.summary_text.insert("1.0", "\n".join(summary_lines))
        self.summary_text.configure(state="disabled")

        # ── Cell grid ──
        # Sort: blasts first, then healthy, then debris
        cells_sorted = sorted(
            [d for d in detections if not d.get('is_debris', False)],
            key=lambda x: not x.get('is_blast', False)
        )
        debris_sorted = [d for d in detections if d.get('is_debris', False)]
        all_sorted = cells_sorted + debris_sorted
        
        for idx, cell in enumerate(all_sorted):
            row = idx // self.grid_columns
            col = idx % self.grid_columns
            
            self.cell_grid_scroll.grid_columnconfigure(col, weight=1)
            
            thumb = CellThumbnail(self.cell_grid_scroll, cell)
            thumb.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

    # ─── IMAGE DISPLAY ──────────────────────────────────
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
        self.img_label.master.update_idletasks()
        frame = self.img_label.master
        max_w = max(frame.winfo_width() - 30, 200)
        max_h = max(frame.winfo_height() - 120, 200)

        ratio = min(max_w / pil_img.width, max_h / pil_img.height)
        new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        ctk_img = ctk.CTkImage(pil_img, size=new_size)
        self.img_label.configure(image=ctk_img, text="")
        self.img_label.image = ctk_img

    def _safe_progress(self, text):
        self.after(0, lambda: self.progress_lbl.configure(text=text))


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    app = ClassificationDemoApp()
    app.mainloop()
