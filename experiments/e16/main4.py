import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from Zh.src.utils.memmap_handler import MemmapHandler
from Zh.experiments.e15.datasets import get_target_from_video_id
from Zh.experiments.e15.helpers import get_subject_id_from_video_id
from Zh.src.utils.file_folder import get_file_paths

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
video_ids = [v for v in video_ids if "__base__" in v]

def pool_normalized(arr, n_percentiles=5):
    """Within-video normalize then pool with temporal dynamics."""
    if arr is None or arr.shape[0] < 2:
        return None
    # Within-video normalization
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True) + 1e-8
    arr  = (arr - mean) / std

    # Static pooling
    percentiles = np.linspace(0, 100, n_percentiles)
    static = np.concatenate([
        arr.mean(axis=0),
        arr.std(axis=0),
        *[np.percentile(arr, p, axis=0) for p in percentiles],
    ])

    # Temporal dynamics
    diff1 = np.diff(arr, axis=0)
    dynamic = np.concatenate([
        np.abs(diff1).mean(axis=0),   # mean absolute velocity
        diff1.std(axis=0),            # movement variance
    ])

    return np.concatenate([static, dynamic])

X_emb, X_blend, X_other = [], [], []
y, groups = [], []

for vid in video_ids:
    try:
        emb_mm  = MemmapHandler(path=vid, mode="r")
        emb     = emb_mm.read(slice=(slice(None), slice(0, 768)))

        feat_basename = os.path.basename(vid).split("__")[0] + "__features.dat"
        feat_path     = os.path.join(FEATURES_DIR, feat_basename)
        feat_mm       = MemmapHandler(path=feat_path, mode="r")
        blend = feat_mm.read(slice=(slice(None), slice(7, 59)))
        other = feat_mm.read(slice=(slice(None), slice(1, 5)))

        if any(a is None or a.shape[0] == 0 for a in [emb, blend, other]):
            print(f"  SKIP: {vid}")
            continue

        X_emb.append(pool_normalized(emb))
        X_blend.append(pool_normalized(blend))
        X_other.append(pool_normalized(other))
        y.append(get_target_from_video_id(vid, DATASET_NAME))
        groups.append(get_subject_id_from_video_id(vid, DATASET_NAME))

    except Exception as e:
        print(f"  ERROR on {vid}: {e}")

X_emb   = np.stack(X_emb)
X_blend = np.stack(X_blend)
X_other = np.stack(X_other)
y       = np.array(y)
groups  = np.array(groups)

logo = LeaveOneGroupOut()

def evaluate(name, X):
    n_components = min(15, X.shape[0] - 1, X.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components)),
        ("model",  RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])),
    ])
    all_preds, all_true = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        all_preds.extend(preds.tolist())
        all_true.extend(y[test_idx].tolist())
    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)
    print(f"  {name:25s}  MAE={mean_absolute_error(all_true, all_preds):.4f}"
          f"  R²={r2_score(all_true, all_preds):.4f}")

# Test modality combinations
evaluate("emb_only",          X_emb)
evaluate("blend_only",        X_blend)
evaluate("other_only",        X_other)
evaluate("blend+other",       np.hstack([X_blend, X_other]))
evaluate("emb+blend",         np.hstack([X_emb,   X_blend]))
evaluate("emb+other",         np.hstack([X_emb,   X_other]))
evaluate("emb+blend+other",   np.hstack([X_emb,   X_blend, X_other]))

# Late fusion: train separate PCA+Ridge per modality, combine predictions
print("\n  -- Late fusion (average predictions) --")
def get_preds(X):
    n_components = min(15, X.shape[0] - 1, X.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components)),
        ("model",  RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])),
    ])
    preds_all, true_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe.fit(X[train_idx], y[train_idx])
        preds_all.extend(pipe.predict(X[test_idx]).tolist())
        true_all.extend(y[test_idx].tolist())
    return np.array(preds_all), np.array(true_all)

preds_emb,   true_ = get_preds(X_emb)
preds_blend, _     = get_preds(X_blend)
preds_other, _     = get_preds(X_other)

for w_blend, w_other in [(0.2, 0.1), (0.3, 0.1), (0.1, 0.05)]:
    w_emb = 1.0 - w_blend - w_other
    fused = w_emb * preds_emb + w_blend * preds_blend + w_other * preds_other
    mae   = mean_absolute_error(true_, fused)
    r2    = r2_score(true_, fused)
    print(f"  emb×{w_emb:.2f} + blend×{w_blend} + other×{w_other}"
          f"  MAE={mae:.4f}  R²={r2:.4f}")

print(f"\n  {'baseline (predict mean)':25s}  MAE={mean_absolute_error(y, np.full_like(y, y.mean())):.4f}")