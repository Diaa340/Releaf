import os, json, numpy as np, pandas as pd
from argparse import ArgumentParser
import yaml

from .io_utils import find_city_npzs, preload_city_data
from .pipeline import run_pipeline_with_params
from .metrics import aggregate_with_gt, aggregate_no_gt

DEFAULT_CFG = {
    "emb_dir": "Releaf_Embeddings_CLIP",
    "best_params": {
        "outlier_frac": 0.2,
        "min_cluster_size": 5,
        "min_samples": 5,
        "sim_thresh_merge": 0.7,
        "sim_thresh_outliers": 0.8,
        "sim_thresh_reassign": 0.7,
        "N_ITER": 2
    },
    "ideal_n_per_city": {
        "seattle_embeddings": 287,
        "calgary_embeddings": 41,
        "columbus_embeddings": 91,
        "san_francisco_embeddings": 652,
        "washington_embeddings": 325,
        "denver_embeddings": 254,
        "new_york_embeddings": 112,
        "la_embeddings": 140
    },
    "gpu": "2"
}

def evaluate_repro(cfg, write_log=print):
    emb_dir = cfg["emb_dir"]
    params = cfg["best_params"]
    ideal_n = cfg["ideal_n_per_city"]
    if cfg.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    write_log(f"[repro] emb_dir={emb_dir}")
    city_npzs = find_city_npzs(emb_dir)
    preload = preload_city_data(city_npzs, write_log)

    per_area = run_pipeline_with_params(params, preload, ideal_n, mode="repro")
    agg = aggregate_with_gt(per_area)

    result = {
        "mode": "repro",
        "per_area_n": len(per_area),
        **agg
    }
    print(json.dumps(result, indent=2))
    return result

def evaluate_deploy(cfg, write_log=print):
    emb_dir = cfg["emb_dir"]
    params = cfg["best_params"]
    ideal_n = cfg["ideal_n_per_city"]
    if cfg.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    write_log(f"[deploy] emb_dir={emb_dir}")
    city_npzs = find_city_npzs(emb_dir)
    preload = preload_city_data(city_npzs, write_log)

    per_area = run_pipeline_with_params(params, preload, ideal_n, mode="deploy")
    agg = aggregate_no_gt(per_area)

    result = {
        "mode": "deploy",
        "per_area_n": len(per_area),
        **agg
    }
    print(json.dumps(result, indent=2))
    return result

def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (optional)")
    ap.add_argument("--mode", choices=["repro", "deploy"], default="repro")
    args = ap.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = DEFAULT_CFG

    if args.mode == "repro":
        return evaluate_repro(cfg)
    else:
        return evaluate_deploy(cfg)
