"""
LHS search procedure (separate from main evaluation).
Run:  python -m releaf_tuning.lhs_search_procedure --samples 100
"""
import os, json, numpy as np, pandas as pd, yaml, traceback
from argparse import ArgumentParser
from pyDOE2 import lhs

from .io_utils import find_city_npzs, preload_city_data
from .pipeline import run_pipeline_with_params
from .metrics import aggregate_with_gt  # LHS is for reproducibility only

def scale_lhs_samples(bounds: dict, lhs_samples: np.ndarray):
    names = list(bounds.keys()); cfgs = []
    for row in lhs_samples:
        c = {}
        for i, name in enumerate(names):
            low, high = bounds[name]
            val = low + (high - low) * row[i]
            if name in ["min_cluster_size", "min_samples", "N_ITER"]:
                val = int(round(val))
            c[name] = val
        cfgs.append(c)
    return cfgs

def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--samples", type=int, default=100)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    emb_dir = cfg["emb_dir"]
    ideal_n = cfg["ideal_n_per_city"]
    param_bounds = cfg["param_bounds"]

    city_npzs = find_city_npzs(emb_dir)
    preload = preload_city_data(city_npzs, print)

    dims = len(param_bounds)
    grid = lhs(dims, samples=args.samples)
    configs = scale_lhs_samples(param_bounds, grid)

    rows = []
    for i, c in enumerate(configs, 1):
        try:
            print(f"Running {i}/{len(configs)}: {json.dumps(c)}")
            per_area = run_pipeline_with_params(c, preload, ideal_n, mode="repro")
            agg = aggregate_with_gt(per_area)
            rows.append({**c, **agg})
        except Exception:
            print(traceback.format_exc())

    df = pd.DataFrame(rows)
    out_csv = "lhs_hyperparam_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

if __name__ == "__main__":
    main()
