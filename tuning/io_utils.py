import os
import numpy as np
from tqdm import tqdm

def find_city_npzs(emb_dir):
    return [os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.endswith("_with_areaids.npz") or f.endswith("_embeddings.npz")]

def preload_city_data(city_npzs, write_log):
    preload = {}
    for cityfile in tqdm(city_npzs, desc="Loading NPZs"):
        try:
            data = np.load(cityfile, allow_pickle=True)
            item = {
                "area_ids": data["area_ids"] if "area_ids" in data else None,
                "img_embs": data["img_embs"],
                "loc_embs": data["loc_embs"],
            }
            if "labels" in data:
                item["labels"] = data["labels"]
            preload[cityfile] = item
        except Exception:
            import traceback
            write_log(f"Error loading {cityfile}: {traceback.format_exc()}")
    return preload
