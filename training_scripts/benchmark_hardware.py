import subprocess
import os
import glob
import pandas as pd
import time
from datetime import datetime

# Configuration Grid
GRID = {
    16: [1, 2, 3, 4, 5, 6, 7, 8],
    32: [1, 2, 3, 4, 5, 6],
    64: [1, 2, 3],
    128: [0, 1]
}

OUTPUT_ROOT = "outputs/benchmarks"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

results = []

def run_benchmark(batch_size, num_workers):
    run_id = f"bench_b{batch_size}_w{num_workers}"
    print(f"\n>>> TESTING: Batch={batch_size}, Workers={num_workers} | RunID: {run_id}")
    
    cmd = [
        "python", "training_scripts/train_base.py",
        "--model", "mnv3l",
        "--fold", "1",
        "--run_name", run_id,
        "--output_root", OUTPUT_ROOT,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--epochs", "3",
        "--no_live"
    ]
    
    try:
        subprocess.run(cmd, capture_output=False, text=True, timeout=600)
        
        # train_base.py creates: OUTPUT_ROOT/run_id/run_id_fold1_TIMESTAMP_metrics.csv
        # Glob for it since we don't know the exact timestamp
        run_dir = os.path.join(OUTPUT_ROOT, run_id)
        csv_files = glob.glob(os.path.join(run_dir, "*_metrics.csv"))
        
        if csv_files:
            # Use the most recent one
            metrics_file = max(csv_files, key=os.path.getmtime)
            df = pd.read_csv(metrics_file)
            if len(df) >= 2:
                e1_time = df.iloc[0]['epoch_time']
                e2_time = df.iloc[1]['epoch_time']
                print(f"SUCCESS: E1={e1_time}s, E2={e2_time}s")
                return {
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "epoch_1": e1_time,
                    "epoch_2": e2_time,
                    "status": "Success"
                }
        
        print("WARNING: Metrics file incomplete or missing.")
        return {"batch_size": batch_size, "num_workers": num_workers, "status": "Failed (No Metrics)"}
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Batch={batch_size}, Workers={num_workers} exceeded 10 min limit.")
        return {"batch_size": batch_size, "num_workers": num_workers, "status": "Timeout (600s)"}
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"batch_size": batch_size, "num_workers": num_workers, "status": f"Error: {str(e)}"}

# Start Benchmarking
print("="*60)
print(" AUTOMATED HARDWARE BENCHMARK SUITE")
print("="*60)

for bs, workers_list in GRID.items():
    for nw in workers_list:
        res = run_benchmark(bs, nw)
        results.append(res)
        
        # Save intermediate results
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(os.path.join(OUTPUT_ROOT, "benchmark_summary.csv"), index=False)

# Final Markdown Summary
print("\n" + "="*60)
print(" FINAL RESULTS")
print("="*60)
summary_table = pd.DataFrame(results)
print(summary_table.to_markdown(index=False))

with open(os.path.join(OUTPUT_ROOT, "benchmark_summary.md"), "w") as f:
    f.write("# Automated Hardware Benchmark Results\n\n")
    f.write(summary_table.to_markdown(index=False))
