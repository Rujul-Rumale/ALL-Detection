"""
ALL Detection System - Local UI
CustomTkinter-based interface for blood smear analysis
"""

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import threading
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.stage1_screening import ALLScreener
from detection.blast_detector_v5 import detect_blasts


# === THEME ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class CellCard(ctk.CTkFrame):
    """Card widget for displaying a detected cell with its metrics."""
    
    def __init__(self, parent, cell_data, crop_image=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        is_blast = cell_data.get('is_blast', False)
        border_color = "#FF4444" if is_blast else "#44AA44"
        
        self.configure(
            corner_radius=10,
            border_width=2,
            border_color=border_color,
            fg_color=("#2B2B2B", "#2B2B2B")
        )
        
        # Layout: [Image] [Metrics] [Explanation]
        self.grid_columnconfigure(1, weight=1)
        
        # Cell thumbnail
        if crop_image is not None:
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(crop_image)
            pil_img = pil_img.resize((80, 80), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(pil_img, size=(80, 80))
            img_label = ctk.CTkLabel(self, image=ctk_img, text="")
            img_label.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        else:
            # Placeholder
            placeholder = ctk.CTkLabel(self, text="🔬", font=("Arial", 40))
            placeholder.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        
        # Cell ID and classification
        cls_text = cell_data['classification']
        cls_color = "#FF6666" if is_blast else "#66AA66"
        
        header = ctk.CTkLabel(
            self,
            text=f"Cell #{cell_data['id']} - {cls_text}",
            font=("Arial", 14, "bold"),
            text_color=cls_color
        )
        header.grid(row=0, column=1, sticky="w", padx=5, pady=(10, 0))
        
        # Metrics (V5 per-cell classification)
        m_text = f"Circ: {cell_data['circularity']*100:.0f}% | Homo: {cell_data['homogeneity']*100:.0f}% | Score: {cell_data['score']:.2f}"
        # NOTE: TFLite is now image-level, shown in banner, not per-cell
        
        metrics = ctk.CTkLabel(
            self,
            text=m_text,
            font=("Arial", 11),
            text_color="#AAAAAA",
            justify="left"
        )
        metrics.grid(row=1, column=1, sticky="w", padx=5, pady=(0, 10))


class ALLDetectionApp(ctk.CTk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.title("ALL Detection System")
        self.geometry("1200x700")
        self.minsize(900, 500)
        
        # Initialize models
        self.screener = None
        self.current_result = None
        
        self._create_ui()
        self._load_models()
    
    def _create_ui(self):
        """Create the main UI layout."""
        
        # Top bar
        self.top_bar = ctk.CTkFrame(self, height=50)
        self.top_bar.pack(fill="x", padx=10, pady=5)
        
        self.title_label = ctk.CTkLabel(
            self.top_bar,
            text="🔬 ALL Detection System",
            font=("Arial", 18, "bold")
        )
        self.title_label.pack(side="left", padx=10)
        
        self.load_btn = ctk.CTkButton(
            self.top_bar,
            text="📂 Load Image",
            command=self._load_image,
            width=120
        )
        self.load_btn.pack(side="right", padx=10)
        
        self.status_label = ctk.CTkLabel(
            self.top_bar,
            text="Ready",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.status_label.pack(side="right", padx=20)
        
        # Main content area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Configure grid with weights favoring results
        self.main_frame.grid_columnconfigure(0, weight=2, minsize=300)  # Image panel (40%)
        self.main_frame.grid_columnconfigure(1, weight=3, minsize=400)  # Results panel (60%)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Image display
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.image_frame.grid_propagate(False)  # Prevent image from expanding frame
        
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="Load an image to begin analysis",
            font=("Arial", 14),
            text_color="#666666"
        )
        self.image_label.pack(expand=True)
        
        # Right panel - Results
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Result banner
        self.result_banner = ctk.CTkLabel(
            self.results_frame,
            text="No analysis yet",
            font=("Arial", 16, "bold"),
            height=40
        )
        self.result_banner.pack(fill="x", padx=10, pady=10)
        
        # Scrollable cell list
        self.cell_list = ctk.CTkScrollableFrame(
            self.results_frame,
            label_text="Detected Cells"
        )
        self.cell_list.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Bottom bar
        self.bottom_bar = ctk.CTkFrame(self, height=30)
        self.bottom_bar.pack(fill="x", padx=10, pady=5)
        
        self.progress_label = ctk.CTkLabel(
            self.bottom_bar,
            text="",
            font=("Arial", 10),
            text_color="#666666"
        )
        self.progress_label.pack(side="left", padx=10)
    
    def _load_models(self):
        """Load screening model in background."""
        def load():
            try:
                self.screener = ALLScreener()
                self.after(0, lambda: self.status_label.configure(text="Models loaded ✓"))
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(text=f"Model error: {e}"))
        
        threading.Thread(target=load, daemon=True).start()
    
    def _load_image(self):
        """Open file dialog and analyze selected image."""
        filepath = filedialog.askopenfilename(
            title="Select Blood Smear Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self._analyze_image(filepath)
    
    def _analyze_image(self, filepath):
        """Run analysis pipeline: Detection -> Animation -> LLM."""
        
        def run_pipeline():
            try:
                # Update UI
                self.after(0, lambda: self.status_label.configure(text="Analyzing..."))
                self.after(0, lambda: self.result_banner.configure(text="Processing...", text_color="#FFAA00"))
                self.after(0, lambda: self._display_image(filepath)) # Reset image
                
                # --- STAGE 1: Detection (V5) ---
                self.after(0, lambda: self.progress_label.configure(text="Stage 1: Detecting cells..."))
                detection = detect_blasts(filepath, return_crops=True, return_all_cells=True)
                
                if not detection['detections']:
                    self.after(0, lambda: self._show_detection_results(filepath, detection, None))
                    return

                # --- STAGE 1.5: TFLite Screening on FULL IMAGE ---
                # NOTE: Model was trained on full microscope images, NOT cell crops
                self.after(0, lambda: self.progress_label.configure(text="Stage 2: TFLite screening..."))
                
                try:
                    full_image_result = self.screener.predict(filepath)
                    detection['tflite_result'] = full_image_result
                    detection['tflite_positive_count'] = len(detection['detections']) if full_image_result['positive'] else 0
                    
                    # Propagate full-image result to all cells for display consistency
                    for cell in detection['detections']:
                        cell['tflite_positive'] = full_image_result['positive']
                        cell['tflite_confidence'] = full_image_result['all_probability']
                except Exception as e:
                    print(f"TFLite Error: {e}")
                    detection['tflite_positive_count'] = 0

                # --- STAGE 2: Animation & LLM ---
                
                # Start LLM in background if there are blasts
                blasts = [c for c in detection['detections'] if c.get('is_blast')]
                self.llm_response = None
                self.llm_thread = None
                
                if blasts:
                    self.llm_thread = threading.Thread(target=self._run_llm_background, args=(blasts,))
                    self.llm_thread.start()
                
                # Start Animation in Main Thread (scheduled)
                self.after(0, lambda: self._start_animation(filepath, detection))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self.status_label.configure(text=f"Error: {e}"))
        
        threading.Thread(target=run_pipeline, daemon=True).start()

    def _run_llm_background(self, blasts):
        """Run LLM in background thread."""
        try:
            self.llm_response = self._generate_explanation(blasts)
        except Exception:
            self.llm_response = None

    def _start_animation(self, filepath, detection):
        """Start drawing boxes one by one."""
        self.progress_label.configure(text="Visualizing detections...")
        
        # Sort cells: Blasts first!
        sorted_cells = sorted(detection['detections'], key=lambda x: not x.get('is_blast', False))
        
        # Load clean image
        import cv2
        original_img = cv2.imread(filepath)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        self._animate_step(original_img, sorted_cells, 0, detection)

    def _animate_step(self, current_img, cells, index, full_detection):
        """Recursive animation step."""
        if index >= len(cells):
            # Animation done - waiting for LLM?
            self.progress_label.configure(text="Finalizing analysis...")
            self._check_llm_and_finish(full_detection)
            return
            
        # Draw NEXT box
        cell = cells[index]
        x, y, w, h = cell['bbox']
        color = (255, 0, 0) if cell.get('is_blast', False) else (0, 255, 0)
        
        import cv2
        # Draw on copy to keep history
        cv2.rectangle(current_img, (x, y), (x+w, y+h), color, 3) # Thicker box
        
        # Show updated image
        self._display_array_image(current_img)
        
        # Update progress
        status_text = f"Detected: {cell['classification']} (Score: {cell['score']})"
        self.progress_label.configure(text=status_text)
        
        # Schedule next step (300ms delay)
        self.after(500, lambda: self._animate_step(current_img, cells, index + 1, full_detection))

    def _check_llm_and_finish(self, detection):
        """Wait for LLM thread to finish then show results."""
        if self.llm_thread and self.llm_thread.is_alive():
            self.progress_label.configure(text="Generating AI summary...")
            self.after(200, lambda: self._check_llm_and_finish(detection))
            return
            
        # LLM done (or wasn't needed)
        self._show_detection_results(detection['image'], detection, self.llm_response)
    
    def _generate_explanation(self, blasts):
        """Generate dynamic clinical summary based on actual cell metrics."""
        try:
            from datetime import datetime
            
            n_blasts = len(blasts)
            avg_circ = sum(b['circularity'] for b in blasts) / n_blasts
            avg_homo = sum(b['homogeneity'] for b in blasts) / n_blasts
            avg_score = sum(b['score'] for b in blasts) / n_blasts
            
            # Dynamic explanation based on metrics
            parts = []
            
            # Circularity interpretation
            if avg_circ > 0.75:
                parts.append(f"The cells show high circularity ({avg_circ*100:.0f}%), indicating round nuclear contours typical of immature blast cells.")
            elif avg_circ > 0.5:
                parts.append(f"Moderate circularity ({avg_circ*100:.0f}%) suggests some nuclear irregularity.")
            else:
                parts.append(f"Low circularity ({avg_circ*100:.0f}%) indicates irregular nuclear borders.")
            
            # Homogeneity interpretation
            if avg_homo > 0.85:
                parts.append(f"Homogeneity is elevated ({avg_homo*100:.0f}%), reflecting fine chromatin pattern consistent with lymphoblasts.")
            elif avg_homo > 0.75:
                parts.append(f"Moderate homogeneity ({avg_homo*100:.0f}%) suggests somewhat uniform chromatin distribution.")
            
            # Score interpretation
            if avg_score > 3.2:
                parts.append(f"The high aggregate score ({avg_score:.2f}) strongly suggests L1-type lymphoblasts.")
            elif avg_score > 2.8:
                parts.append(f"Score ({avg_score:.2f}) is borderline; further examination recommended.")
            
            # Combine
            content = " ".join(parts)
            if not content:
                content = f"Detected {n_blasts} suspected blast cell(s) with average score {avg_score:.2f}."
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            return f"[{timestamp} AI] {content}"
        except Exception as e:
            return f"[Error] Could not generate summary: {e}"
    
    def _show_healthy_result(self, filepath, screening):
        """Display healthy result."""
        self.status_label.configure(text="Analysis complete")
        self.progress_label.configure(text="")
        
        # Load and display image
        self._display_image(filepath)
        
        # Update banner
        confidence = screening['confidence'] * 100
        self.result_banner.configure(
            text=f"✅ HEALTHY ({confidence:.1f}% confidence)",
            text_color="#44AA44"
        )
        
        # Clear cell list
        for widget in self.cell_list.winfo_children():
            widget.destroy()
        
        info_label = ctk.CTkLabel(
            self.cell_list,
            text="No blast cells detected.\nImage classified as healthy.",
            font=("Arial", 12),
            text_color="#888888"
        )
        info_label.pack(pady=20)
    
    def _show_detection_results(self, filepath, detection, explanation=None):
        """Display detection results using V5 classification (no TFLite screening)."""
        self.status_label.configure(text="Analysis complete")
        
        blast_count = detection['blast_count']
        total_cells = detection['total_cells']
        tflite_pos = detection.get('tflite_positive_count', 0)
        
        self.progress_label.configure(text=f"Found {total_cells} cells, {blast_count} blast(s) (TF confirmed: {tflite_pos})")
        
        # Display annotated image
        if 'annotated_image' in detection:
            self._display_array_image(detection['annotated_image'])
        else:
            self._display_image(filepath)
        
        # Update banner based on TFLite (image-level) AND V5 (cell-level)
        tflite_result = detection.get('tflite_result', {})
        tflite_conf = tflite_result.get('confidence', 0) * 100 if tflite_result else 0
        tflite_positive = tflite_result.get('positive', False) if tflite_result else False
        
        if tflite_positive or blast_count > 0:
            # Show combined result
            banner_text = f"⚠️ SUSPECTED ALL"
            if tflite_result:
                banner_text += f" (TFLite: {tflite_conf:.0f}%)"
            if blast_count > 0:
                banner_text += f" - {blast_count} Blast(s)"
            self.result_banner.configure(text=banner_text, text_color="#FF4444")
        else:
            banner_text = f"✅ HEALTHY"
            if tflite_result:
                banner_text += f" (TFLite: {tflite_conf:.0f}%)"
            banner_text += f" - {total_cells} cells analyzed"
            self.result_banner.configure(text=banner_text, text_color="#44AA44")
        
        # Clear and populate cell list
        for widget in self.cell_list.winfo_children():
            widget.destroy()
            
        # Display AI Summary if available
        if explanation:
            summary_frame = ctk.CTkFrame(self.cell_list, fg_color="#333333", corner_radius=8)
            summary_frame.pack(fill="x", padx=5, pady=(0, 10))
            
            title = ctk.CTkLabel(summary_frame, text="🤖 AI Summary", font=("Roboto Mono", 14, "bold"))
            title.pack(anchor="w", padx=10, pady=(10, 5))
            
            text = ctk.CTkLabel(summary_frame, text=explanation, font=("Roboto Mono", 13), 
                               wraplength=350, justify="left", text_color="#FFFFFF")
            text.pack(anchor="w", padx=10, pady=(0, 10))
        
        if not detection['detections']:
            info_label = ctk.CTkLabel(
                self.cell_list,
                text="No cells detected in image.",
                font=("Arial", 12),
                text_color="#888888"
            )
            info_label.pack(pady=20)
            return
        
        # Sort: blasts first
        cells = sorted(detection['detections'], key=lambda x: not x.get('is_blast', False))
        
        for cell in cells:
            crop = cell.get('crop', None)
            card = CellCard(self.cell_list, cell, crop_image=crop)
            card.pack(fill="x", padx=5, pady=5)
    
    def _display_image(self, filepath):
        """Display image from file path."""
        try:
            pil_img = Image.open(filepath)
            self._display_pil_image(pil_img)
        except Exception as e:
            self.image_label.configure(text=f"Error loading image: {e}")
    
    def _display_array_image(self, img_array):
        """Display image from numpy array (RGB)."""
        try:
            pil_img = Image.fromarray(img_array)
            self._display_pil_image(pil_img)
        except Exception as e:
            self.image_label.configure(text=f"Error displaying image: {e}")
    
    def _display_pil_image(self, pil_img):
        """Display PIL image, scaling to fit panel."""
        # Get available size
        self.image_frame.update()
        
        # Cap size to avoid explosion (45% max)
        max_w = min(self.image_frame.winfo_width(), self.winfo_width() * 0.45)
        max_h = min(self.image_frame.winfo_height(), self.winfo_height() * 0.8)
        
        if max_w < 200 or max_h < 200:
            max_w, max_h = 400, 400
            
        max_w = int(max_w - 20)
        max_h = int(max_h - 20)
        
        # Scale image
        img_w, img_h = pil_img.size
        scale = min(max_w / img_w, max_h / img_h)
        new_size = (int(img_w * scale), int(img_h * scale))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Display
        ctk_img = ctk.CTkImage(pil_img, size=new_size)
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img  # Keep reference


def main():
    app = ALLDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
