import argparse, glob, os, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="Root containing run/*/eval/fixed_points/summary.csv")
ap.add_argument("--out", required=True, help="Output CSV path")
args = ap.parse_args()

hits = sorted(glob.glob(os.path.join(args.root, "**/run/eval/fixed_points/summary.csv"), recursive=True))
if not hits:
    print("No per-run summary.csv files found in", args.root)
    sys.exit(0)

dfs = []
for p in hits:
    run_eval = os.path.dirname(os.path.dirname(p)) 
    run_dir  = os.path.dirname(run_eval)   
    out_dir  = os.path.dirname(run_dir)    

    df = pd.read_csv(p)
    df.insert(0, "summary_path", p)
    df.insert(1, "run_dir", run_dir)
    df.insert(2, "cfg_dir", out_dir)
    dfs.append(df)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
pd.concat(dfs, ignore_index=True).to_csv(args.out, index=False)
print(f"Wrote {len(dfs)} runs → {args.out}")
