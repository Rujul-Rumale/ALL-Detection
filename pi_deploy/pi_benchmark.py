#!/usr/bin/env python3
"""
pi_benchmark.py
===============
Standalone TFLite inference benchmark for Raspberry Pi 5.
No project imports required — fully self-contained.

Dependencies (all available on Pi via piwheels or pip):
  tflite-runtime, numpy, psutil

Usage:
  python3 pi_benchmark.py \
      --model models/tflite_final/mnv3l_fold1_best.tflite \
      --n_runs 100 \
      --out results/pi_benchmark.csv
"""

import argparse
import csv
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import psutil

# ── tflite-runtime import (Pi does NOT have full tensorflow) ──────────────────
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback: allow running on a development machine with full TF installed
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("[INFO] tflite_runtime not found; using tensorflow.lite.Interpreter")
    except ImportError:
        print("[ERROR] Neither tflite_runtime nor tensorflow is installed.")
        sys.exit(1)

# ── ImageNet normalisation ────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── CSV columns ───────────────────────────────────────────────────────────────
CSV_HEADER = [
    "timestamp", "hostname", "model_path", "model_name",
    "input_shape", "input_dtype", "n_runs",
    "latency_mean_ms", "latency_std_ms",
    "latency_min_ms",  "latency_max_ms",
    "latency_p50_ms",  "latency_p95_ms",
    "model_size_mb", "process_rss_mb",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TFLite inference benchmark for Pi 5")
    p.add_argument(
        "--model", required=True,
        help="Path to .tflite model file",
    )
    p.add_argument(
        "--n_runs", type=int, default=100,
        help="Number of benchmark inference runs (default: 100)",
    )
    p.add_argument(
        "--out", default="results/pi_benchmark.csv",
        help="CSV output path (rows are appended; created with header if absent)",
    )
    p.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup runs before timing (default: 10)",
    )
    return p.parse_args()


def make_input(input_shape: tuple) -> np.ndarray:
    """
    Generate one random float32 input tensor in the expected input_shape,
    ImageNet-normalised (channel-wise mean / std applied).

    For the standard model: shape = [1, 3, 320, 320] (NCHW).
    """
    rng = np.random.default_rng(seed=0)
    # Random pixel values uniformly in [0, 1] — channel-first
    c, h, w = input_shape[1], input_shape[2], input_shape[3]
    img = rng.random((h, w, c), dtype=np.float32)              # HWC
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))                         # CHW
    return np.expand_dims(img, axis=0).astype(np.float32)     # NCHW


def run_benchmark(args: argparse.Namespace) -> dict:
    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    model_size_mb = os.path.getsize(model_path) / (1024 ** 2)
    model_name    = os.path.splitext(os.path.basename(model_path))[0]

    # ── Load interpreter ──────────────────────────────────────────────────────
    print(f"\nLoading: {model_name}")
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()

    in_det   = interp.get_input_details()[0]
    out_det  = interp.get_output_details()[0]
    in_shape = tuple(in_det["shape"])
    in_dtype = str(in_det["dtype"])

    print(f"  Input  : shape={in_shape}  dtype={in_dtype}")
    print(f"  Output : shape={tuple(out_det['shape'])}  dtype={out_det['dtype']}")
    print(f"  Model size: {model_size_mb:.2f} MB")

    # ── Prepare input ─────────────────────────────────────────────────────────
    input_tensor = make_input(in_shape)

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"\n  Warmup ({args.warmup} runs) …", end="", flush=True)
    for _ in range(args.warmup):
        interp.set_tensor(in_det["index"], input_tensor)
        interp.invoke()
    print(" done.")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print(f"  Benchmarking ({args.n_runs} runs) …", end="", flush=True)
    latencies_ms = []
    for _ in range(args.n_runs):
        t0 = time.perf_counter()
        interp.set_tensor(in_det["index"], input_tensor)
        interp.invoke()
        _ = interp.get_tensor(out_det["index"])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    print(" done.")

    latencies = np.array(latencies_ms, dtype=np.float64)
    stats = {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms":  float(np.std(latencies, ddof=1)),
        "latency_min_ms":  float(np.min(latencies)),
        "latency_max_ms":  float(np.max(latencies)),
        "latency_p50_ms":  float(np.percentile(latencies, 50)),
        "latency_p95_ms":  float(np.percentile(latencies, 95)),
    }

    # ── System info ───────────────────────────────────────────────────────────
    rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)

    record = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "hostname":        socket.gethostname(),
        "model_path":      model_path,
        "model_name":      model_name,
        "input_shape":     str(in_shape),
        "input_dtype":     in_dtype,
        "n_runs":          args.n_runs,
        "model_size_mb":   round(model_size_mb, 3),
        "process_rss_mb":  round(rss_mb, 2),
        **{k: round(v, 3) for k, v in stats.items()},
    }
    return record


def append_csv(out_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    print(f"\n  Results appended → {out_path}")


def print_summary(record: dict) -> None:
    bar = "─" * 44
    print(f"\n{bar}")
    print(f"  {'Model:':<18} {record['model_name']}")
    print(f"  {'Input:':<18} {record['input_shape']}  {record['input_dtype']}")
    print(f"  {'Runs:':<18} {record['n_runs']}")
    print(bar)
    print(f"  {'Mean latency:':<18} {record['latency_mean_ms']:.1f} ms")
    print(f"  {'Std:':<18} ±{record['latency_std_ms']:.1f} ms")
    print(f"  {'Min:':<18} {record['latency_min_ms']:.1f} ms")
    print(f"  {'Max:':<18} {record['latency_max_ms']:.1f} ms")
    print(f"  {'P50 latency:':<18} {record['latency_p50_ms']:.1f} ms")
    print(f"  {'P95 latency:':<18} {record['latency_p95_ms']:.1f} ms")
    print(bar)
    print(f"  {'Model size:':<18} {record['model_size_mb']:.2f} MB")
    print(f"  {'RSS memory:':<18} {record['process_rss_mb']:.1f} MB")
    print(f"  {'Host:':<18} {record['hostname']}")
    print(bar)


if __name__ == "__main__":
    args   = parse_args()
    record = run_benchmark(args)
    print_summary(record)
    append_csv(args.out, record)
