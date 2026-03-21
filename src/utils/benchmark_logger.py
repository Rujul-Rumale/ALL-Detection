"""
benchmark_logger.py
===================
Hardware-aware benchmarking logger for the Pi 5 deployment tests.

Logs per-run telemetry to a daily CSV in outputs/pi_benchmarks/.
Gracefully handles non-Pi environments (vcgencmd calls wrapped in try/except).

Usage (called by DemoPipeline):
    logger = BenchmarkLogger()
    logger.start_session(image_filename, model_name, arch, fold)
    ...run pipeline...
    logger.finish_session(results_dict, per_cell_times_list)
"""

import os
import csv
import time
import socket
import subprocess
import numpy as np
import psutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "pi_benchmarks"


class BenchmarkLogger:
    def __init__(self):
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        self._start_time: float = 0.0
        self._image_filename: str = ""
        self._model_name: str = ""
        self._arch: str = ""
        self._fold = ""

    # ── Session control ────────────────────────────────────────────────────────

    def start_session(self, image_filename: str, model_name: str, arch: str, fold):
        """Call immediately before watershed begins."""
        self._image_filename = os.path.basename(image_filename)
        self._model_name     = model_name
        self._arch           = arch
        self._fold           = str(fold)
        self._start_time     = time.perf_counter()

    def finish_session(self, results: dict, per_cell_times: list):
        """
        Call after classify stage completes.

        results dict must contain:
            watershed_time, sam_time, classify_time, total_time  (all in seconds)
            total_cells, blast_count, healthy_count, debris_count

        per_cell_times: list of floats (milliseconds per cell inference)
        """
        # ── Per-cell stats ────────────────────────────────────────────────────
        if per_cell_times:
            arr = np.array(per_cell_times, dtype=np.float64)
            pc_mean = float(arr.mean())
            pc_min  = float(arr.min())
            pc_max  = float(arr.max())
            pc_std  = float(arr.std())
        else:
            pc_mean = pc_min = pc_max = pc_std = 0.0

        # ── System metrics ────────────────────────────────────────────────────
        try:
            process_rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        except Exception:
            process_rss_mb = -1.0

        try:
            cpu_pct = psutil.cpu_percent(interval=0.1)
        except Exception:
            cpu_pct = -1.0

        # ── Pi-specific metrics ───────────────────────────────────────────────
        pi_core_temp   = self._vcgencmd_temp()
        pi_throttled   = self._vcgencmd_throttled()

        # ── Assemble row ──────────────────────────────────────────────────────
        row = {
            "timestamp":           datetime.now().isoformat(timespec="seconds"),
            "hostname":            socket.gethostname(),
            "image_filename":      self._image_filename,
            "model_name":          self._model_name,
            "arch":                self._arch,
            "fold":                self._fold,
            "watershed_ms":        round(results.get("watershed_time", 0) * 1000, 2),
            "sam_ms":              round(results.get("sam_time",       0) * 1000, 2),
            "classify_total_ms":   round(results.get("classify_time",  0) * 1000, 2),
            "per_cell_ms_mean":    round(pc_mean, 3),
            "per_cell_ms_min":     round(pc_min,  3),
            "per_cell_ms_max":     round(pc_max,  3),
            "per_cell_ms_std":     round(pc_std,  3),
            "total_ms":            round(results.get("total_time",     0) * 1000, 2),
            "process_rss_mb":      round(process_rss_mb, 2),
            "cpu_percent":         round(cpu_pct, 1),
            "n_cells":             results.get("total_cells",   0),
            "n_blasts":            results.get("blast_count",   0),
            "n_healthy":           results.get("healthy_count", 0),
            "n_debris":            results.get("debris_count",  0),
            "pi_core_temp_C":      pi_core_temp,
            "pi_throttled":        pi_throttled,
        }

        # ── Write CSV ─────────────────────────────────────────────────────────
        csv_path = BENCHMARK_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_benchmark.csv"
        write_header = not csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # ── Console summary ───────────────────────────────────────────────────
        print(
            f"[Benchmark] {self._model_name} | "
            f"Total: {row['total_ms']:.0f}ms | "
            f"SAM: {row['sam_ms']:.0f}ms | "
            f"Classify: {row['classify_total_ms']:.0f}ms | "
            f"Cells: {row['n_cells']} (B:{row['n_blasts']} H:{row['n_healthy']}) | "
            f"Temp: {pi_core_temp} | "
            f"RSS: {row['process_rss_mb']:.0f}MB → {csv_path.name}"
        )

    # ── Pi helpers ─────────────────────────────────────────────────────────────

    def _vcgencmd_temp(self) -> str:
        """Returns core temp as float string e.g. '52.1', or 'N/A'."""
        try:
            out = subprocess.check_output(
                ["vcgencmd", "measure_temp"],
                timeout=2, stderr=subprocess.DEVNULL
            ).decode().strip()
            # Output: "temp=52.1'C"
            return out.replace("temp=", "").replace("'C", "").strip()
        except Exception:
            return "N/A"

    def _vcgencmd_throttled(self) -> str:
        """Returns throttle hex flags e.g. '0x0', or 'N/A'."""
        try:
            out = subprocess.check_output(
                ["vcgencmd", "get_throttled"],
                timeout=2, stderr=subprocess.DEVNULL
            ).decode().strip()
            # Output: "throttled=0x0"
            return out.replace("throttled=", "").strip()
        except Exception:
            return "N/A"
