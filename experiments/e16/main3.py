import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from Zh.experiments.e15.datasets import get_target_from_video_id
from Zh.experiments.e15.helpers import get_subject_id_from_video_id
from Zh.src.utils.file_folder import get_file_paths
from Zh.src.utils.memmap_handler import MemmapHandler

# EMBEDDINGS_DIR = (
#     "/work/long_lab/Mojtaba/data/DROZY/data/videos/3.1-embedding/dinov2-base/cls/ROI0"
# )
# FEATURES_DIR = "/work/long_lab/Mojtaba/data/DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05/features"

DATASET_NAME = "UTA"
EMBEDDINGS_DIR = (
    "/work/long_lab/Mojtaba/data/UTA/3.1-embedding/dinov2-base/cls/ROI0"
)
FEATURES_DIR = "/work/long_lab/Mojtaba/data/UTA/3-augmented_roi/masked_aligned_cropped0.05/features"

video_ids = get_file_paths(input_path=EMBEDDINGS_DIR, depth=3, EXT_LIST=".dat")
# video_ids = [v for v in video_ids if "__base__" in v]


def pool_modality(arr):
    """Extract mean, std, 10th, 50th, 90th percentiles per feature dimension."""
    if arr is None or arr.shape[0] == 0:
        return None
    return np.concatenate(
        [
            arr.mean(axis=0),
            arr.std(axis=0),
            np.percentile(arr, 10, axis=0),
            np.percentile(arr, 50, axis=0),
            np.percentile(arr, 90, axis=0),
        ]
    )

def pool_modality_normalized(arr):
    """Normalize within the video first, then pool."""
    if arr is None or arr.shape[0] < 2:
        return None
    # Remove subject-specific baseline by z-scoring within the video
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True) + 1e-8
    arr_norm = (arr - mean) / std
    return pool_modality(arr_norm)  # your original pool function

X_emb, X_blend, X_other = [], [], []
y, groups = [], []

for vid in video_ids:
    try:
        # Embeddings
        emb_mm = MemmapHandler(path=vid, mode="r")
        emb = emb_mm.read(slice=(slice(None), slice(0, 768)))

        # Features (blendshapes + other)
        feat_basename = os.path.basename(vid).split("__")[0] + "__features.dat"
        feat_path = os.path.join(FEATURES_DIR, feat_basename)
        feat_mm = MemmapHandler(path=feat_path, mode="r")
        blend = feat_mm.read(slice=(slice(None), slice(7, 59)))  # 52-d blendshapes
        other = feat_mm.read(slice=(slice(None), slice(1, 5)))  # 4-d head movement

        if any(a is None or a.shape[0] == 0 for a in [emb, blend, other]):
            print(f"  SKIP: {vid}")
            continue

        X_emb.append(pool_modality_normalized(emb))
        X_blend.append(pool_modality_normalized(blend))
        X_other.append(pool_modality_normalized(other))
        y.append(get_target_from_video_id(vid, DATASET_NAME))
        groups.append(get_subject_id_from_video_id(vid, DATASET_NAME))

    except Exception as e:
        print(f"  ERROR on {vid}: {e}")

X_emb = np.stack(X_emb)  # (N, 768*5)
X_blend = np.stack(X_blend)  # (N, 52*5)
X_other = np.stack(X_other)  # (N, 4*5)
y = np.array(y)
groups = np.array(groups)

print(f"Shapes — emb: {X_emb.shape}, blend: {X_blend.shape}, other: {X_other.shape}")

# Test different feature combinations
feature_sets = {
    "emb_only": X_emb,
    "blend_only": X_blend,
    "other_only": X_other,
    "blend+other": np.hstack([X_blend, X_other]),
    "emb+blend+other": np.hstack([X_emb, X_blend, X_other]),
}

logo = LeaveOneGroupOut()

for name, X in feature_sets.items():
    # PCA to avoid curse of dimensionality, then Ridge
    n_components = min(20, X.shape[0] - 1, X.shape[1])
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])),
        ]
    )

    all_preds, all_true = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        all_preds.extend(preds.tolist())
        all_true.extend(y[test_idx].tolist())

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)
    print(f"  {name:20s}  MAE={mae:.4f}  R²={r2:.4f}")

print(
    f"\n  {'baseline (predict mean)':20s}  MAE={mean_absolute_error(y, np.full_like(y, y.mean())):.4f}"
)
