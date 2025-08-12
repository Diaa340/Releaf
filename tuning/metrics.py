from collections import Counter
import numpy as np
from sklearn.metrics import v_measure_score

def _freqs(labels):
    freqs = Counter(labels)
    tot = sum(freqs.values())
    return freqs, tot

def shannon_entropy(labels):
    freqs, tot = _freqs(labels)
    return 0.0 if tot == 0 else -sum((c/tot)*np.log(c/tot) for c in freqs.values())

def simpson_index(labels):
    # Gini–Simpson (1 - sum p_i^2); higher = more diverse
    freqs, tot = _freqs(labels)
    return 0.0 if tot == 0 else 1.0 - sum((c/tot)**2 for c in freqs.values())

def richness(labels):
    return len(np.unique(labels))

def adjust_outliers(pred):
    pred_adj = pred.copy()
    next_label = pred_adj.max() + 1 if len(pred_adj) > 0 else 0
    for i in np.where(pred_adj == -1)[0]:
        pred_adj[i] = next_label
        next_label += 1
    return pred_adj

# ---------- WITH ground truth ----------
def metrics_with_gt(true, pred):
    pred_adj = adjust_outliers(pred)
    return dict(
        richness_pred=richness(pred_adj),
        shannon_pred=shannon_entropy(pred_adj),
        simpson_pred=simpson_index(pred_adj),
        vscore=v_measure_score(true, pred_adj),
    )

def aggregate_with_gt(per_area):
    """
    per_area items must include: vscore, shannon_pred, simpson_pred, richness_pred, richness_rmse, area_coverage
    """
    import pandas as pd
    import numpy as np
    if not per_area:
        return {
            "vscore_mean": np.nan,
            "richness_rmse": np.nan,
            "area_coverage": np.nan,
            "shannon_pred_mean": np.nan,
            "simpson_pred_mean": np.nan,
            "richness_pred_mean": np.nan,
        }
    df = pd.DataFrame(per_area)
    weights = df["area_coverage"]
    return {
        "vscore_mean": np.average(df["vscore"], weights=weights),
        "richness_rmse": df["richness_rmse"].mean(),
        "area_coverage": df["area_coverage"].mean(),
        "shannon_pred_mean": df["shannon_pred"].mean(),
        "simpson_pred_mean": df["simpson_pred"].mean(),
        "richness_pred_mean": df["richness_pred"].mean(),
    }

# ---------- NO ground truth ----------
def metrics_no_gt(pred):
    pred_adj = adjust_outliers(pred)
    return dict(
        richness_pred=richness(pred_adj),
        shannon_pred=shannon_entropy(pred_adj),
        simpson_pred=simpson_index(pred_adj),
    )

def aggregate_no_gt(per_area):
    """
    per_area items must include: shannon_pred, simpson_pred, richness_pred, area_coverage
    """
    import pandas as pd
    import numpy as np
    if not per_area:
        return {
            "area_coverage": np.nan,
            "shannon_pred_mean": np.nan,
            "simpson_pred_mean": np.nan,
            "richness_pred_mean": np.nan,
        }
    df = pd.DataFrame(per_area)
    return {
        "area_coverage": df["area_coverage"].mean(),
        "shannon_pred_mean": df["shannon_pred"].mean(),
        "simpson_pred_mean": df["simpson_pred"].mean(),
        "richness_pred_mean": df["richness_pred"].mean(),
    }
