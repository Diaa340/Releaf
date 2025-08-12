import os, random, numpy as np, hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from .postprocess import (
    eliminate_outliers_frac, merge_similar_clusters,
    merge_similar_outliers, reassign_remaining_outliers
)
from .metrics import (
    metrics_with_gt, metrics_no_gt, richness,
)

def _choose_areas(area_ids, n):
    unique = list(set(area_ids)) if area_ids is not None else [0]  # fallback single area if missing
    return random.sample(unique, min(n, len(unique)))

def run_pipeline_with_params(cfg, city_data, ideal_n_per_city, mode="repro"):
    """
    mode: "repro" (expects GT labels in NPZ) or "deploy" (no GT).
    """
    results_all = []
    for cityfile, data in city_data.items():
        city_name = os.path.basename(cityfile).split('_with_areaids.npz')[0].split('_embeddings.npz')[0]
        areas = ideal_n_per_city.get(city_name, 1)
        selected_areas = _choose_areas(data.get("area_ids"), areas)

        for area_id in selected_areas:
            if data.get("area_ids") is not None:
                idxs = np.where(data["area_ids"] == area_id)[0]
            else:
                idxs = np.arange(len(data["img_embs"]))  # whole file as one area
            if len(idxs) == 0:
                continue

            img_embs = data["img_embs"][idxs]
            loc_embs = data["loc_embs"][idxs]
            labels_gt = data.get("labels")
            labels_gt = labels_gt[idxs] if labels_gt is not None else None

            if img_embs.shape[0] < max(2, cfg["min_cluster_size"]):
                continue

            # 1) HDBSCAN on location embeddings
            cosine_dist_loc = 1 - cosine_similarity(loc_embs).astype(np.float64)
            clusterer = hdbscan.HDBSCAN(
                metric='precomputed',
                min_cluster_size=cfg["min_cluster_size"],
                min_samples=cfg["min_samples"]
            )
            labels_current = clusterer.fit_predict(cosine_dist_loc)

            # 2) Post-processing on image embeddings
            for _ in range(cfg["N_ITER"]):
                labels_elim = eliminate_outliers_frac(img_embs, labels_current, cfg["outlier_frac"])
                labels_merged = merge_similar_clusters(img_embs, labels_elim, cfg["sim_thresh_merge"])
                labels_out_merged = merge_similar_outliers(img_embs, labels_merged, cfg["sim_thresh_outliers"])
                labels_final = reassign_remaining_outliers(img_embs, labels_out_merged, cfg["sim_thresh_reassign"])
                labels_current = labels_final

            if mode == "repro" and labels_gt is not None:
                m = metrics_with_gt(labels_gt, labels_current)
                m.update({
                    "richness_rmse": abs(m["richness_pred"] - richness(labels_gt)),
                    "area_coverage": len(selected_areas),
                })
            else:
                m = metrics_no_gt(labels_current)
                m.update({
                    "area_coverage": len(selected_areas),
                })
            results_all.append(m)
    return results_all
